##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_castm.c
        bli_castm.h
        bli_castnzm.c
        bli_castnzm.h
        bli_castv.c
        bli_castv.h
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
