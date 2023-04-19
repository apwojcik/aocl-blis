##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_ind.c
        bli_ind.h
        bli_l3_ind.c
        bli_l3_ind.h
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_subdirectory(cntx)
add_subdirectory(oapi)
add_subdirectory(tapi)
add_subdirectory(ukernels)
