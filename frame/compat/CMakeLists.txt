##Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bla_amax.h
        bla_amin.c
        bla_amin.h
        bla_asum.c
        bla_asum.h
        bla_gemm3m.c
        bla_gemmt.c
        bla_ger.c
        bla_hemm.c
        bla_hemv.c
        bla_her.c
        bla_her2.c
        bla_her2k.c
        bla_herk.c
        bla_nrm2.c
        bla_symm.c
        bla_symv.c
        bla_syr.c
        bla_syr2.c
        bla_syr2k.c
        bla_syrk.c
        bla_trmm.c
        bla_trmv.c
        bla_trsv.c
        bla_gemm_batch.c
        bla_axpby.c
        bla_omatcopy.c
        bla_imatcopy.c
        bla_omatcopy2.c
        bla_omatadd.c
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
            bla_amax_amd.c
            bla_axpy_amd.c
            bla_copy_amd.c
            bla_dot_amd.c
            bla_gemm_amd.c
            bla_gemv_amd.c
            bla_scal_amd.c
            bla_swap_amd.c
            bla_trsm_amd.c
        )
else()
    target_sources(${TARGET_NAME}
        PRIVATE
            bla_amax.c
            bla_axpy.c
            bla_copy.c
            bla_dot.c
            bla_gemm.c
            bla_gemv.c
            bla_scal.c
            bla_swap.c
            bla_trsm.c
        )
endif()
    
add_subdirectory(cblas)
add_subdirectory(check)
add_subdirectory(f2c)
