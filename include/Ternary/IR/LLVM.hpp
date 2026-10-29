#ifndef TERNARY_IR_LLVM_HPP
#define TERNARY_IR_LLVM_HPP
#include "Optimize.hpp"
#include <bit>
#include <iomanip>
#include <sstream>
#include <array>
namespace bt::ir {
// Standalone text lowering: depends only on verified core IR, not frontend AST.
class LLVMEmitter {
    std::ostringstream out;
    std::unordered_map<std::string,std::string> names;
    std::unordered_map<uint32_t,Type> slots;
    size_t serial=0;
    std::string result_type, current_block;
    bool terminated=false, main=false, gpu=false, optimize=false, debug=false;
    static std::string type(Type t) { return t==Type::Void?"void":t==Type::Float64?"double":"i64"; }
    static std::string quote(const std::string& s) {
        std::ostringstream o;
        for(unsigned char c:s) {
            if(c=='"' || c=='\\' || c<32 || c>=127) o<<'\\'<<std::uppercase<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(c);
            else o<<c;
        }
        return o.str();
    }
    std::string reg() { return "%r"+std::to_string(++serial); }
    std::string label() { return "b"+std::to_string(++serial); }
    std::string instruction(const std::string& value) { auto r=reg();out<<"  "<<r<<" = "<<value<<(debug && main?", !dbg !7":"")<<"\n";return r; }
    void block(const std::string& name) {out<<name<<":\n";current_block=name;terminated=false;}
    void jump(const std::string& target) {out<<"  br label %"<<target<<"\n";terminated=true;}
    void guard(const std::string& bad) {
        auto ok=label();out<<"  br i1 "<<bad<<", label %error, label %"<<ok<<"\n";block(ok);
    }
    std::string compare(const std::string& value,Type t,bool positive) {
        return instruction(std::string(t==Type::Float64?"fcmp ":"icmp ")+
            (t==Type::Float64?(positive?"ogt":"olt"):(positive?"sgt":"slt"))+" "+type(t)+" "+value+", "+(t==Type::Float64?"0.0":"0"));
    }
    void check64(const std::string& n,int64_t low=math::ScalarInteger::minimum,int64_t high=math::ScalarInteger::maximum) {
        auto lo=instruction("icmp slt i64 "+n+", "+std::to_string(low));
        auto hi=instruction("icmp sgt i64 "+n+", "+std::to_string(high));guard(instruction("or i1 "+lo+", "+hi));
    }
    std::string checked128(const std::string& n) {
        auto lo=instruction("icmp slt i128 "+n+", "+std::to_string(math::ScalarInteger::minimum));
        auto hi=instruction("icmp sgt i128 "+n+", "+std::to_string(math::ScalarInteger::maximum));
        guard(instruction("or i1 "+lo+", "+hi));return instruction("trunc i128 "+n+" to i64");
    }
    std::string arithmetic(const Expr& e,const std::string& a,const std::string& b) {
        std::string op=e.op==Op::Add?"add":e.op==Op::Sub?"sub":"mul";
        if(e.type==Type::Float64) return instruction("f"+op+" double "+a+", "+b);
        if(optimize && fits(integer_range(e))) return instruction(op+" i64 "+a+", "+b);
        if(gpu) {
            // +/- of two int39 inputs fit i64. Check multiplication BEFORE it
            // executes; no overflowing device operation or i128 device ABI.
            if(e.op==Op::Mul) {
                auto an=instruction("icmp slt i64 "+a+", 0"),bn=instruction("icmp slt i64 "+b+", 0");
                auto na=instruction("sub i64 0, "+a),nb=instruction("sub i64 0, "+b);
                auto aa=instruction("select i1 "+an+", i64 "+na+", i64 "+a);
                auto ab=instruction("select i1 "+bn+", i64 "+nb+", i64 "+b);
                auto zero=instruction("icmp eq i64 "+aa+", 0");
                auto divisor=instruction("select i1 "+zero+", i64 1, i64 "+aa);
                auto bound=instruction("udiv i64 "+std::to_string(math::ScalarInteger::maximum)+", "+divisor);
                guard(instruction("icmp ugt i64 "+ab+", "+bound));
            }
            auto r=instruction(op+" i64 "+a+", "+b);check64(r);return r;
        }
        auto x=instruction("sext i64 "+a+" to i128"),y=instruction("sext i64 "+b+" to i128");
        return checked128(instruction(op+" i128 "+x+", "+y));
    }
    std::string expression(const Expr& e) {
        if(e.op==Op::Integer) return std::to_string(e.integer);
        if(e.op==Op::Floating) {
            std::ostringstream s;s<<"0x"<<std::uppercase<<std::hex<<std::setw(16)<<std::setfill('0')<<std::bit_cast<uint64_t>(e.floating);return s.str();
        }
        if(e.op==Op::Parameter) return instruction("load "+type(e.type)+", ptr %v"+std::to_string(e.symbol));
        if(e.op==Op::SignSelect) {
            auto condition=expression(*e.operands[0]);auto p=label(),other=label(),z=label(),n=label(),merge=label();
            auto cmp=compare(condition,e.operands[0]->type,true);
            out<<"  br i1 "<<cmp<<", label %"<<p<<", label %"<<other<<"\n";
            block(other);cmp=compare(condition,e.operands[0]->type,false);
            out<<"  br i1 "<<cmp<<", label %"<<n<<", label %"<<z<<"\n";
            std::array<std::string,3> labels={p,z,n},values,ends;
            for(size_t i=0;i<3;++i) {block(labels[i]);values[i]=expression(*e.operands[i+1]);ends[i]=current_block;jump(merge);}
            block(merge);
            if(e.type==Type::Void) return {};
            return instruction("phi "+type(e.type)+" [ "+values[0]+", %"+ends[0]+" ], [ "+values[1]+", %"+ends[1]+" ], [ "+values[2]+", %"+ends[2]+" ]");
        }
        std::vector<std::string> args;
        for(auto& a:e.operands) args.push_back(expression(*a)); // explicit left-to-right effects
        if(e.op==Op::Call) {
            auto depth=instruction("load i32, ptr @bt.core.depth");guard(instruction("icmp uge i32 "+depth+", 512"));
            auto next=instruction("add i32 "+depth+", 1");out<<"  store i32 "<<next<<", ptr @bt.core.depth\n";
            std::string call="call "+type(e.type)+" @"+names.at(e.name)+"(";
            for(size_t i=0;i<args.size();++i) call+=(i?", ":"")+type(e.operands[i]->type)+" "+args[i];call+=")";
            std::string value;
            if(e.type==Type::Void) out<<"  "<<call<<(debug && main?", !dbg !7":"")<<"\n";else value=instruction(call);
            out<<"  store i32 "<<depth<<", ptr @bt.core.depth\n";
            auto status=instruction("load i32, ptr @bt.core.error");guard(instruction("icmp ne i32 "+status+", 0"));return value;
        }
        if(e.op==Op::Print) {out<<"  call i32 (ptr, ...) @printf(ptr @bt.core.fmt."<<(e.operands[0]->type==Type::Float64?"f":"i")<<", "<<type(e.operands[0]->type)<<" "<<args[0]<<")"<<(debug && main?", !dbg !7":"")<<"\n";return {};}
        const auto& a=args.at(0);
        if(e.op==Op::IntCast) {if(e.type==Type::Trit) check64(a,-1,1);return a;}
        if(e.op==Op::IntToFloat) return instruction("sitofp i64 "+a+" to double");
        if(e.op==Op::Negate) return instruction(e.type==Type::Float64?"fneg double "+a:"sub i64 0, "+a);
        if(e.op==Op::Abs) {
            if(e.type==Type::Float64) return instruction("call double @llvm.fabs.f64(double "+a+")");
            auto neg=instruction("sub i64 0, "+a),cmp=instruction("icmp slt i64 "+a+", 0");
            return instruction("select i1 "+cmp+", i64 "+neg+", i64 "+a);
        }
        if(e.op==Op::Sqrt) return instruction("call double @llvm.sqrt.f64(double "+a+")");
        const auto& b=args.at(1);
        if(e.op==Op::Fma) return instruction("call double @llvm.fma.f64(double "+a+", double "+b+", double "+args[2]+")");
        if(e.op==Op::Add || e.op==Op::Sub || e.op==Op::Mul) return arithmetic(e,a,b);
        if(e.op==Op::DigitAnd || e.op==Op::DigitOr || e.op==Op::DigitXor)
            return instruction("call i64 @bt.core.logic(i64 "+a+", i64 "+b+", i32 "+std::to_string(e.op==Op::DigitAnd?0:e.op==Op::DigitOr?1:2)+")");
        if(e.op==Op::Intrinsic) {
            const auto& c=args[2];const auto& d=args[3];
            auto pp=instruction("and i64 "+a+", "+c),nn=instruction("and i64 "+b+", "+d);
            if(e.name=="planar_dot64") {
                auto pn=instruction("and i64 "+a+", "+d),np=instruction("and i64 "+b+", "+c);
                auto positives=instruction("or i64 "+pp+", "+nn),negatives=instruction("or i64 "+pn+", "+np);
                auto pc=instruction("call i64 @llvm.ctpop.i64(i64 "+positives+")"),nc=instruction("call i64 @llvm.ctpop.i64(i64 "+negatives+")");
                return instruction("sub i64 "+pc+", "+nc);
            }
            std::string r=e.name=="btha_carry_p"?pp:nn;
            if(e.name=="btha_sum_p" || e.name=="btha_sum_n") {
                bool p=e.name=="btha_sum_p";
                auto x=instruction("xor i64 "+(p?a:b)+", "+(p?c:d));
                auto mask=instruction("or i64 "+(p?b:a)+", "+(p?d:c));mask=instruction("xor i64 "+mask+", -1");
                r=instruction("and i64 "+x+", "+mask);r=instruction("or i64 "+r+", "+(p?nn:pp));
            }
            check64(r);return r;
        }
        bool floating=e.operands[0]->type==Type::Float64;
        auto comparison=[&](const std::string& fp,const std::string& ip) {
            return instruction((floating?"fcmp "+fp:"icmp "+ip)+" "+type(e.operands[0]->type)+" "+a+", "+b);
        };
        if(e.op==Op::Spaceship) {
            auto gt=comparison("ogt","sgt"),lt=comparison("olt","slt");
            auto v=instruction("select i1 "+gt+", i64 1, i64 0");return instruction("select i1 "+lt+", i64 -1, i64 "+v);
        }
        std::string condition;
        switch(e.op) {
        case Op::Equal:condition=comparison("oeq","eq");break;case Op::NotEqual:condition=comparison("une","ne");break;
        case Op::Less:condition=comparison("olt","slt");break;case Op::Greater:condition=comparison("ogt","sgt");break;
        case Op::LessEqual:condition=comparison("ole","sle");break;case Op::GreaterEqual:condition=comparison("oge","sge");break;
        default:throw std::invalid_argument("unsupported scalar LLVM operation");
        }
        return instruction("select i1 "+condition+", i64 1, i64 -1");
    }
    void collect(const std::vector<Statement>& body) {
        for(auto& s:body) {if(s.kind==StatementKind::Let) slots.emplace(s.symbol,s.type);collect(s.body);collect(s.alternative);collect(s.negative);}
    }
    void statements(const std::vector<Statement>& body) {
        for(auto& s:body) {
            if(terminated) break;
            switch(s.kind) {
            case StatementKind::Let:case StatementKind::Assign: {
                auto value=expression(*s.value);out<<"  store "<<type(s.type)<<" "<<value<<", ptr %v"<<s.symbol<<"\n";break;
            }
            case StatementKind::Evaluate:expression(*s.value);break;
            case StatementKind::Block:statements(s.body);break;
            case StatementKind::Return: {
                auto value=s.value?expression(*s.value):std::string{};
                if(gpu) {out<<"  store i64 "<<value<<", ptr %output.lane\n  store i32 0, ptr %error.lane\n  ret void\n";}
                else {if(main) value=instruction("trunc i64 "+value+" to i32");out<<"  ret "<<result_type<<(result_type=="void"?"":" "+value)<<"\n";}
                terminated=true;break;
            }
            case StatementKind::While: {
                auto cond=label(),bodyLabel=label(),done=label();jump(cond);block(cond);
                auto value=expression(*s.value),cmp=compare(value,s.value->type,true);
                out<<"  br i1 "<<cmp<<", label %"<<bodyLabel<<", label %"<<done<<"\n";
                block(bodyLabel);statements(s.body);if(!terminated) jump(cond);block(done);break;
            }
            case StatementKind::If:case StatementKind::SignSelect: {
                auto value=expression(*s.value),p=label(),z=label(),n=label(),other=label(),done=label();
                auto cmp=compare(value,s.value->type,true);out<<"  br i1 "<<cmp<<", label %"<<p<<", label %"<<(s.kind==StatementKind::If?z:other)<<"\n";
                if(s.kind==StatementKind::SignSelect) {block(other);cmp=compare(value,s.value->type,false);out<<"  br i1 "<<cmp<<", label %"<<n<<", label %"<<z<<"\n";}
                bool all=true;
                auto branch=[&](const std::string& name,const std::vector<Statement>& b) {block(name);statements(b);if(!terminated){all=false;jump(done);}};
                branch(p,s.body);branch(z,s.alternative);if(s.kind==StatementKind::SignSelect) branch(n,s.negative);
                if(all) terminated=true;else block(done);break;
            }
            }
        }
    }
    static std::string helpers() {
        return R"IR(
declare i64 @llvm.ctpop.i64(i64)
define internal i64 @bt.core.logic(i64 %a0, i64 %b0, i32 %mode) {
entry:
  br label %loop
loop:
  %a = phi i64 [ %a0, %entry ], [ %qa, %body ]
  %b = phi i64 [ %b0, %entry ], [ %qb, %body ]
  %acc = phi i64 [ 0, %entry ], [ %next.acc, %body ]
  %place = phi i64 [ 1, %entry ], [ %next.place, %body ]
  %both = or i64 %a, %b
  %done = icmp eq i64 %both, 0
  br i1 %done, label %exit, label %body
body:
  %ar = srem i64 %a, 3
  %br = srem i64 %b, 3
  %aq = sdiv i64 %a, 3
  %bq = sdiv i64 %b, 3
  %ap = icmp eq i64 %ar, 2
  %an = icmp eq i64 %ar, -2
  %bp = icmp eq i64 %br, 2
  %bn = icmp eq i64 %br, -2
  %aqp = add i64 %aq, 1
  %aqn = sub i64 %aq, 1
  %bqp = add i64 %bq, 1
  %bqn = sub i64 %bq, 1
  %qa0 = select i1 %ap, i64 %aqp, i64 %aq
  %qa = select i1 %an, i64 %aqn, i64 %qa0
  %qb0 = select i1 %bp, i64 %bqp, i64 %bq
  %qb = select i1 %bn, i64 %bqn, i64 %qb0
  %ra0 = select i1 %ap, i64 -1, i64 %ar
  %ra = select i1 %an, i64 1, i64 %ra0
  %rb0 = select i1 %bp, i64 -1, i64 %br
  %rb = select i1 %bn, i64 1, i64 %rb0
  %less = icmp slt i64 %ra, %rb
  %min = select i1 %less, i64 %ra, i64 %rb
  %max = select i1 %less, i64 %rb, i64 %ra
  %product = mul i64 %ra, %rb
  %is.and = icmp eq i32 %mode, 0
  %is.or = icmp eq i32 %mode, 1
  %d0 = select i1 %is.and, i64 %min, i64 %product
  %digit = select i1 %is.or, i64 %max, i64 %d0
  %weighted = mul i64 %digit, %place
  %next.acc = add i64 %acc, %weighted
  %next.place = mul i64 %place, 3
  br label %loop
exit:
  ret i64 %acc
}
)IR";
    }
    void function(const Function& f,bool entry=false) {
        main=entry;result_type=entry?"i32":type(f.result);slots.clear();
        for(auto p:f.parameters) slots.emplace(p.symbol,p.type);collect(f.body);
        out<<"define "<<(entry?"":"internal ")<<result_type<<" @"<<(entry?"bt_main":names.at(f.name))<<"(";
        for(size_t i=0;i<f.parameters.size();++i) out<<(i?", ":"")<<type(f.parameters[i].type)<<" %arg"<<f.parameters[i].symbol;
        out<<")"<<(entry && debug?" !dbg !4":"")<<" {\n";block("entry");
        for(auto [id,t]:slots) out<<"  %v"<<id<<" = alloca "<<type(t)<<", align 8\n";
        for(auto p:f.parameters) out<<"  store "<<type(p.type)<<" %arg"<<p.symbol<<", ptr %v"<<p.symbol<<"\n";
        if(entry) out<<"  store i32 0, ptr @bt.core.error\n  store i32 0, ptr @bt.core.depth\n";
        statements(f.body);
        if(!terminated) out<<"  ret "<<result_type<<(result_type=="void"?"":result_type=="double"?" 0.0":" 0")<<"\n";
        block("error");out<<"  store i32 1, ptr @bt.core.error\n  ret "<<result_type<<(result_type=="void"?"":result_type=="double"?" 0.0":entry?" 1":" 0")<<"\n}\n";
    }
    static void gpu_expression(const Expr& e) {
        if(!integral(e.type) || e.op==Op::Call || e.op==Op::Packed || e.op==Op::Print)
            throw std::invalid_argument("CUDA map profile requires pure integral operations, no calls/IO");
        for(auto& x:e.operands) gpu_expression(*x);
    }
    static void gpu_statements(const std::vector<Statement>& body) {
        for(auto& s:body) {
            if(s.kind==StatementKind::While) throw std::invalid_argument("CUDA map profile does not support loops");
            if(s.value) gpu_expression(*s.value);
            gpu_statements(s.body);gpu_statements(s.alternative);gpu_statements(s.negative);
        }
    }
public:
    std::string generate(const Program& input,const std::string& triple,const std::string& layout,
                         bool optimized=false,bool debugInfo=false,const std::string& filename="program.bt") {
        verify_program(input);Program p=optimized?Optimizer{}.run(input):input;
        out.str("");out.clear();names.clear();serial=0;gpu=false;optimize=optimized;debug=debugInfo;
        out<<"; verified Trium core IR -> LLVM, ABI 1\nsource_filename = \""<<quote(filename)<<"\"\n"
           <<"target triple = \""<<quote(triple)<<"\"\ntarget datalayout = \""<<quote(layout)<<"\"\n"
           <<"@bt.core.error = internal thread_local global i32 0\n@bt.core.depth = internal thread_local global i32 0\n"
           <<"@bt.core.fmt.i = private constant [6 x i8] c\"%lld\\0A\\00\"\n@bt.core.fmt.f = private constant [7 x i8] c\"%.17g\\0A\\00\"\n"
           <<"declare i32 @printf(ptr, ...)\ndeclare double @llvm.fma.f64(double, double, double)\ndeclare double @llvm.sqrt.f64(double)\ndeclare double @llvm.fabs.f64(double)\n"<<helpers();
        out<<"define i32 @bt_core_abi_version() { ret i32 1 }\ndefine i32 @bt_core_status() { %s = load i32, ptr @bt.core.error\n ret i32 %s }\n";
        for(size_t i=0;i<p.functions.size();++i) names.emplace(p.functions[i].name,"bt.core.fn."+std::to_string(i));
        for(auto& f:p.functions) function(f);
        function(Function{"",Type::Int39,{},p.body},true);
        if(debug) {
            out<<"!llvm.dbg.cu = !{!0}\n!llvm.module.flags = !{!2, !3}\n"
               <<"!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: \"Trium core\", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)\n"
               <<"!1 = !DIFile(filename: \""<<quote(filename)<<"\", directory: \".\")\n"
               <<"!2 = !{i32 2, !\"Dwarf Version\", i32 4}\n!3 = !{i32 2, !\"Debug Info Version\", i32 3}\n"
               <<"!4 = distinct !DISubprogram(name: \"bt_main\", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)\n"
               <<"!5 = !DISubroutineType(types: !6)\n!6 = !{}\n!7 = !DILocation(line: 1, column: 1, scope: !4)\n";
        }
        return out.str();
    }
    std::string generate_cuda(const Program& p,const std::string& name,bool optimized=false) {
        verify_program(p);
        auto found=std::find_if(p.functions.begin(),p.functions.end(),[&](auto& f){return f.name==name;});
        if(found==p.functions.end() || !p.body.empty() || found->parameters.size()!=1 ||
           found->parameters[0].type!=Type::Int39 || found->result!=Type::Int39)
            throw std::invalid_argument("CUDA map requires a function (int)->int and no top-level execution");
        gpu_statements(found->body);
        Function f=*found;
        if(optimized) {auto q=Optimizer{}.run(p);f=*std::find_if(q.functions.begin(),q.functions.end(),[&](auto& v){return v.name==name;});}
        out.str("");out.clear();serial=0;slots.clear();gpu=true;optimize=optimized;main=false;debug=false;collect(f.body);
        slots.emplace(f.parameters[0].symbol,Type::Int39);
        out<<"target triple = \"nvptx64-nvidia-cuda\"\n"<<helpers()
           <<"declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()\ndeclare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\ndeclare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()\n"
           <<"define void @bt_kernel(ptr %input, ptr %output, ptr %errors, i64 %count) {\n";
        block("entry");for(auto [id,t]:slots) out<<"  %v"<<id<<" = alloca i64, align 8\n";
        auto tid=instruction("call i32 @llvm.nvvm.read.ptx.sreg.tid.x()"),cta=instruction("call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()"),nt=instruction("call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()");
        auto t=instruction("zext i32 "+tid+" to i64"),c=instruction("zext i32 "+cta+" to i64"),w=instruction("zext i32 "+nt+" to i64");
        auto base=instruction("mul i64 "+c+", "+w),idx=instruction("add i64 "+base+", "+t),inside=instruction("icmp ult i64 "+idx+", %count");
        out<<"  br i1 "<<inside<<", label %active, label %done\n";block("done");out<<"  ret void\n";block("active");
        auto in=instruction("getelementptr i64, ptr %input, i64 "+idx);
        out<<"  %output.lane = getelementptr i64, ptr %output, i64 "<<idx<<"\n  %error.lane = getelementptr i32, ptr %errors, i64 "<<idx<<"\n";
        auto x=instruction("load i64, ptr "+in);check64(x);
        out<<"  store i64 "<<x<<", ptr %v"<<f.parameters[0].symbol<<"\n";statements(f.body);
        if(!terminated) throw std::invalid_argument("CUDA function must return on all paths");
        block("error");out<<"  store i32 1, ptr %error.lane\n  ret void\n}\n!nvvm.annotations = !{!0}\n!0 = !{ptr @bt_kernel, !\"kernel\", i32 1}\n";
        return out.str();
    }
};
} // namespace bt::ir
#endif
