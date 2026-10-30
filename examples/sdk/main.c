#include <Ternary/SDK/sdk.h>
#include <stdio.h>
#include <string.h>
int main(void){bt_sdk_handle context=0,module=0;char error[4096];const char* source="func transform(x:int)->int{return x+1;}";int64_t input[]={-1,0,1},output[3];
 if(bt_sdk_context_create(NULL,&context,error,sizeof error)||bt_sdk_compile(context,source,strlen(source),&module,error,sizeof error)||bt_sdk_map(module,"transform",BT_SDK_CPU,input,3,output,NULL,error,sizeof error)){fprintf(stderr,"%s\n",error);return 1;}
 bt_sdk_float_value nine,root;double binary;
 if(bt_sdk_float_from_ratio(40,9,1,0,&nine,error,sizeof error)||bt_sdk_float_apply(BT_SDK_FP_SQRT,&nine,NULL,NULL,&root,error,sizeof error)||bt_sdk_float_to_double(&root,&binary,error,sizeof error)||binary!=3.0)return 2;
 bt_sdk_release(module,NULL,0);bt_sdk_release(context,NULL,0);return output[0]!=0||output[1]!=1||output[2]!=2;}
