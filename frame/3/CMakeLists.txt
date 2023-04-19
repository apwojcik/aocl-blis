##Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_l3.h
        bli_l3_blocksize.c
        bli_l3_blocksize.h
        bli_l3_check.c
        bli_l3_check.h
        bli_l3_cntl.c
        bli_l3_cntl.h
        bli_l3_direct.c
        bli_l3_direct.h
        bli_l3_oapi.c
        bli_l3_oapi.h
        bli_l3_oapi_ba.c
        bli_l3_oapi_ex.c
        bli_l3_oft.h
        bli_l3_oft_var.h
        bli_l3_packm.c
        bli_l3_packm.h
        bli_l3_prune.c
        bli_l3_prune.h
        bli_l3_sup.c
        bli_l3_sup.h
        bli_l3_sup_packm_a.c
        bli_l3_sup_packm_a.h
        bli_l3_sup_packm_b.c
        bli_l3_sup_packm_b.h
        bli_l3_sup_packm_var.c
        bli_l3_sup_packm_var.h
        bli_l3_sup_ref.c
        bli_l3_sup_ref.h
        bli_l3_sup_var12.c
        bli_l3_sup_var1n2m.c
        bli_l3_tapi.c
        bli_l3_tapi.h
        bli_l3_tapi_ba.c
        bli_l3_tapi_ex.c
        bli_l3_thrinfo.c
        bli_l3_thrinfo.h
        bli_l3_ukr_fpa.c
        bli_l3_ukr_fpa.h
        bli_l3_ukr_oapi.c
        bli_l3_ukr_oapi.h
        bli_l3_ukr_tapi.c
        bli_l3_ukr_tapi.h
        bli_l3_smart_threading.c
        bli_l3_smart_threading.h
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
            bli_l3_sup_int_amd.c
            bli_l3_sup_int.h
        )
else()
    target_sources(${TARGET_NAME}
        PRIVATE
            bli_l3_sup_int.c
            bli_l3_sup_int.h
        )
endif()

add_subdirectory(gemm)
add_subdirectory(hemm)
add_subdirectory(her2k)
add_subdirectory(herk)
add_subdirectory(symm)
add_subdirectory(syr2k)
add_subdirectory(syrk)
add_subdirectory(trmm)
add_subdirectory(trmm3)
add_subdirectory(trsm)
add_subdirectory(gemmt)
