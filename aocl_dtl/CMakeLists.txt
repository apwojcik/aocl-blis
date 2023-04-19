##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved. ##

target_sources(${TARGET_NAME}
     PRIVATE
        aocldtl.c
        aocldtl.h
        aocldtl_blis.c
        aocldtl_blis.h
        aocldtlcf.h
        aoclfal.c
        aoclfal.h
        aoclflist.c
        aoclflist.h
        aoclos.c
        aoclos.h
        aocltpdef.h
     )

target_include_directories(${TARGET_NAME}
     PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
     )
