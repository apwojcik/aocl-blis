##Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.##

set(__compile_definitions
        -DBLAS="AOCL"
        -DN_REPEAT=1000
        -DINT_FS="%lld"
        -DUINT_FS="%llu")

add_executable(BenchAmaxv bench_amaxv.c)
target_link_libraries(BenchAmaxv PRIVATE ${LIB_NAME})
if(ENABLE_OPENMP)
    target_link_libraries(BenchAmaxv PRIVATE OpenMP::OpenMP_C)
endif()
target_compile_definitions(BenchAmaxv PRIVATE ${__compile_definitions})

add_executable(BenchAxpbyv bench_axpbyv.c)
target_link_libraries(BenchAxpbyv PRIVATE ${LIB_NAME})
if(ENABLE_OPENMP)
    target_link_libraries(BenchAxpbyv PRIVATE OpenMP::OpenMP_C)
endif()
target_compile_definitions(BenchAxpbyv PRIVATE ${__compile_definitions})

add_executable(BenchCopyv bench_copyv.c)
target_link_libraries(BenchCopyv PRIVATE ${LIB_NAME})
if(ENABLE_OPENMP)
    target_link_libraries(BenchCopyv PRIVATE OpenMP::OpenMP_C)
endif()
target_compile_definitions(BenchCopyv PRIVATE ${__compile_definitions})

add_executable(BenchDotv bench_dotv.c)
target_link_libraries(BenchDotv PRIVATE ${LIB_NAME})
if(ENABLE_OPENMP)
    target_link_libraries(BenchDotv PRIVATE OpenMP::OpenMP_C)
endif()
target_compile_definitions(BenchDotv PRIVATE ${__compile_definitions})

add_executable(BenchGemm bench_gemm.c)
target_link_libraries(BenchGemm PRIVATE ${LIB_NAME})
if(ENABLE_OPENMP)
    target_link_libraries(BenchGemm PRIVATE OpenMP::OpenMP_C)
endif()
target_compile_definitions(BenchGemm PRIVATE ${__compile_definitions})

add_executable(BenchGemmt bench_gemmt.c)
target_link_libraries(BenchGemmt PRIVATE ${LIB_NAME})
if(ENABLE_OPENMP)
    target_link_libraries(BenchGemmt PRIVATE OpenMP::OpenMP_C)
endif()
target_compile_definitions(BenchGemmt PRIVATE ${__compile_definitions})

add_executable(BenchGemv bench_gemv.c)
target_link_libraries(BenchGemv PRIVATE ${LIB_NAME})
if(ENABLE_OPENMP)
    target_link_libraries(BenchGemv PRIVATE OpenMP::OpenMP_C)
endif()
target_compile_definitions(BenchGemv PRIVATE ${__compile_definitions})

add_executable(BenchGer bench_ger.c)
target_link_libraries(BenchGer PRIVATE ${LIB_NAME})
if(ENABLE_OPENMP)
    target_link_libraries(BenchGer PRIVATE OpenMP::OpenMP_C)
endif()
target_compile_definitions(BenchGer PRIVATE ${__compile_definitions})

add_executable(BenchScalv bench_scalv.c)
target_link_libraries(BenchScalv PRIVATE ${LIB_NAME})
if(ENABLE_OPENMP)
    target_link_libraries(BenchScalv PRIVATE OpenMP::OpenMP_C)
endif()
target_compile_definitions(BenchScalv PRIVATE ${__compile_definitions})

add_executable(BenchSwapv bench_swapv.c)
target_link_libraries(BenchSwapv PRIVATE ${LIB_NAME})
if(ENABLE_OPENMP)
    target_link_libraries(BenchSwapv PRIVATE OpenMP::OpenMP_C)
endif()
target_compile_definitions(BenchSwapv PRIVATE ${__compile_definitions})

add_executable(BenchSyrk bench_syrk.c)
target_link_libraries(BenchSyrk PRIVATE ${LIB_NAME})
if(ENABLE_OPENMP)
    target_link_libraries(BenchSyrk PRIVATE OpenMP::OpenMP_C)
endif()
target_compile_definitions(BenchSyrk PRIVATE ${__compile_definitions})

add_executable(BenchTrsm bench_trsm.c)
target_link_libraries(BenchTrsm PRIVATE ${LIB_NAME})
if(ENABLE_OPENMP)
    target_link_libraries(BenchTrsm PRIVATE OpenMP::OpenMP_C)
endif()
target_compile_definitions(BenchTrsm PRIVATE ${__compile_definitions})

add_executable(BenchTrsv bench_trsv.c)
target_link_libraries(BenchTrsv PRIVATE ${LIB_NAME})
if(ENABLE_OPENMP)
    target_link_libraries(BenchTrsv PRIVATE OpenMP::OpenMP_C)
endif()
target_compile_definitions(BenchTrsv PRIVATE ${__compile_definitions})
