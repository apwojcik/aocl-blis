##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_l2.h
        bli_l2_check.c
        bli_l2_check.h
        bli_l2_fpa.c
        bli_l2_fpa.h
        bli_l2_ft.h
        bli_l2_ft_unb.h
        bli_l2_ker.h
        bli_l2_ker_prot.h
        bli_l2_oapi.c
        bli_l2_oapi.h
        bli_l2_oapi_ba.c
        bli_l2_oapi_ex.c
        bli_l2_tapi.c
        bli_l2_tapi.h
        bli_l2_tapi_ba.c
        bli_l2_tapi_ex.c
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_subdirectory(gemv)
add_subdirectory(ger)
add_subdirectory(hemv)
add_subdirectory(her)
add_subdirectory(her2)
add_subdirectory(symv)
add_subdirectory(syr)
add_subdirectory(syr2)
add_subdirectory(trmv)
add_subdirectory(trsv)

