##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved. ##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_l1v.h
        bli_l1v_check.c
        bli_l1v_check.h
        bli_l1v_fpa.c
        bli_l1v_fpa.h
        bli_l1v_ft.h
        bli_l1v_ft_ker.h
        bli_l1v_ker.h
        bli_l1v_ker_prot.h
        bli_l1v_oapi.c
        bli_l1v_oapi.h
        bli_l1v_oapi_ba.c
        bli_l1v_oapi_ex.c
        bli_l1v_tapi.c
        bli_l1v_tapi.h
        bli_l1v_tapi_ba.c
        bli_l1v_tapi_ex.c
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
