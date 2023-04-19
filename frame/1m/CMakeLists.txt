##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_l1m.h
        bli_l1m_check.c
        bli_l1m_check.h
        bli_l1m_fpa.c
        bli_l1m_fpa.h
        bli_l1m_ft.h
        bli_l1m_ft_ker.h
        bli_l1m_ker.h
        bli_l1m_ker_prot.h
        bli_l1m_oapi.c
        bli_l1m_oapi.h
        bli_l1m_oapi_ba.c
        bli_l1m_oapi_ex.c
        bli_l1m_oft_var.h
        bli_l1m_tapi.c
        bli_l1m_tapi.h
        bli_l1m_tapi_ba.c
        bli_l1m_tapi_ex.c
        bli_l1m_unb_var1.c
        bli_l1m_unb_var1.h
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_subdirectory(packm)
add_subdirectory(unpackm)
