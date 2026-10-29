#include "include/Ternary/Math/Integer.hpp"
#include "include/Ternary/IR/Core.hpp"
#include <iostream>
#include <random>
#include <string>
#include <stdexcept>

#define CHECK(c) do { if (!(c)) throw std::runtime_error("check failed: " #c); } while (false)
template<class E, class F> void rejects(F f) {
    bool failed = false; try { f(); } catch (const E&) { failed = true; } CHECK(failed);
}
static void mathematics() {
    using namespace bt::math;
    using I = ScalarInteger;
    static_assert(I::maximum == 2026277576509488133LL);
    static_assert((BalancedInteger<1>(1) + BalancedInteger<1>(-1)).value() == 0);
    rejects<std::overflow_error>([] { I(I::maximum) + I(1); });
    rejects<std::overflow_error>([] { I(I::minimum) - I(1); });
    rejects<std::overflow_error>([] { I(I::maximum) * I(2); });
    rejects<std::domain_error>([] { I(1) / I(0); });
    rejects<std::overflow_error>([] { divmod_euclidean(INT64_MIN,-1); });
    rejects<std::overflow_error>([] { round_quotient(INT64_MIN,-1); });
    CHECK(divmod_euclidean(-7,3).quotient == -3);
    CHECK(divmod_euclidean(-7,-3).quotient == 3);
    CHECK(divmod_euclidean(-7,-3).remainder == 2);
    CHECK(round_quotient(1,2) == 1 && round_quotient(-1,2) == -1);
    CHECK(round_quotient(1,-2) == -1 && round_quotient(-1,-2) == 1);
    std::mt19937_64 rng(0x334d415448);
    for (int i = 0; i < 10000; ++i) {
        const int64_t a = static_cast<int64_t>(rng() % (2*I::maximum+1)) - I::maximum;
        const int64_t b = static_cast<int64_t>(rng() % (2*I::maximum+1)) - I::maximum;
        if (!b) continue;
        auto [q,r] = divmod_euclidean(a,b);
        CHECK(static_cast<__int128>(b)*q+r == a);
        CHECK(r >= 0 && r < (b < 0 ? -b : b));
        CHECK((I(a) / I(b)).value() == q && (I(a) % I(b)).value() == r);
    }
}
static void typed_ir() {
    using namespace bt::ir;
    auto a = std::make_shared<Expr>(Expr{Op::Integer, Type::Int39, {}}); a->integer = 7;
    auto b = std::make_shared<Expr>(Expr{Op::Integer, Type::Trit, {}}); b->integer = 1;
    Expr sum{Op::Add, Type::Int39, {a,b}};
    CHECK(verify(sum) == Effect::MayFail);
    Expr conversion{Op::IntToFloat, Type::Float64, {a}};
    CHECK(verify(conversion) == Effect::Pure);
    rejects<std::invalid_argument>([&] { Expr e{Op::Add, Type::Float64, {a,b}}; verify(e); });
    rejects<std::invalid_argument>([&] { Expr e{Op::Add, Type::Int39, {a}}; verify(e); });
    rejects<std::invalid_argument>([] { Expr e{Op::Packed, Type::Packed64, {}}; e.positive=e.negative=1; verify(e); });
    rejects<std::invalid_argument>([] { Expr e{Op::Integer, Type::Trit, {}}; e.integer=2; verify(e); });
    rejects<std::invalid_argument>([] { Expr e{Op::Parameter, Type::Int39, {}}; verify(e); });
    Expr select{Op::SignSelect, Type::Int39, {b,a,a,a}};
    CHECK(verify(select) == Effect::Pure);
    rejects<std::invalid_argument>([&] { Expr e{Op::SignSelect, Type::Float64, {b,a,a,a}}; verify(e); });
}
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--oracle") {
            int64_t a,b;
            while (std::cin >> a >> b) {
                try {
                    auto [q,r] = bt::math::divmod_euclidean(a,b);
                    auto rounded = bt::math::round_quotient(a,b);
                    std::cout << q << ' ' << r << ' ' << rounded << '\n';
                } catch (const std::domain_error&) { std::cout << "domain\n"; }
                  catch (const std::overflow_error&) { std::cout << "overflow\n"; }
            }
        } else { mathematics(); typed_ir(); std::cout << "Core mathematics and typed IR verifier passed\n"; }
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
