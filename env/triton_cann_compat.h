// 给 triton-ascend 编译 kernel 启动代码时用的补丁头(由 env/triton_cxx.sh 用 -include 带上)。
//
// torch_npu 2.14 的 AclInterface.h 声明了三个用到 CANN 9.x 才有的类型的函数:
//   AclmdlRICondHandleCreate / AclmdlRICondHandleGetCondPtr / AclmdlRIAddCondTask
// CANN 8.x 的头文件里没有 aclmdlRICondHandle / aclmdlRICondTaskParams,于是凡是
// include 了 torch_npu 头文件的代码在 CANN 8.x 上都编不过。
//
// 启动代码只是 include 那个头,**并不调用**这三个函数,所以这里只需要让声明能通过
// 语法检查:一个不透明句柄,一个不完整的结构体(函数声明里按值出现不要求完整类型)。
// 只有 CANN 头里确实没有这两个名字时,triton_cxx.sh 才会带上本文件。
#ifndef QWEN36_TRITON_CANN_COMPAT_H
#define QWEN36_TRITON_CANN_COMPAT_H
typedef void *aclmdlRICondHandle;
struct aclmdlRICondTaskParams;
#endif
