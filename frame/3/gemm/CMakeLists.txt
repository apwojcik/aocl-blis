##Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_gemm.h
        bli_gemm_blk_var1.c
        bli_gemm_blk_var2.c
        bli_gemm_blk_var3.c
        bli_gemm_cntl.c
        bli_gemm_cntl.h
        bli_gemm_int.c
        bli_gemm_int.h
        bli_gemm_ker_var1.c
        bli_gemm_ker_var2.c
        bli_gemm_ker_var2_md.c
        bli_gemm_md.c
        bli_gemm_md.h
        bli_gemm_md_c2r_ref.c
        bli_gemm_md_c2r_ref.h
        bli_gemm_packab.c
        bli_gemm_var.h
        bli_gemm_front.h
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

# Select AMD specific sources for AMD configurations.
if(${TARGET_ARCH} STREQUAL zen OR 
   ${TARGET_ARCH} STREQUAL zen2 OR
   ${TARGET_ARCH} STREQUAL zen3 OR
   ${TARGET_ARCH} STREQUAL zen4 OR
   ${TARGET_ARCH} STREQUAL amdzen)
    target_sources(${TARGET_NAME}
        PRIVATE
            bli_gemm_front_amd.c
        )
else()
    target_sources(${TARGET_NAME}
        PRIVATE
            bli_gemm_front.c
        )
endif()

add_subdirectory(ind)
