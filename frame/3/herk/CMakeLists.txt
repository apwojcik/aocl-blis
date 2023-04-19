##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_herk.h
        bli_herk_front.c
        bli_herk_front.h
        bli_herk_l_ker_var2.c
        bli_herk_u_ker_var2.c
        bli_herk_var.h
        bli_herk_x_ker_var2.c
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
