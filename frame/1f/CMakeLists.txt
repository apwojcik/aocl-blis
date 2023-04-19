##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
     PRIVATE
        bli_l1f.h
        bli_l1f_check.c
        bli_l1f_check.h
        bli_l1f_fpa.c
        bli_l1f_fpa.h
        bli_l1f_ft.h
        bli_l1f_ft_ker.h
        bli_l1f_ker.h
        bli_l1f_ker_prot.h
        bli_l1f_oapi.c
        bli_l1f_oapi.h
        bli_l1f_oapi_ba.c
        bli_l1f_oapi_ex.c
        bli_l1f_tapi.c
        bli_l1f_tapi.h
        bli_l1f_tapi_ba.c
        bli_l1f_tapi_ex.c
     )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
