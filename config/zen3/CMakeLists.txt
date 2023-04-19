##Copyright (C) 2020, Advanced Micro Devices, Inc ##

target_sources(${TARGET_NAME}
    PRIVATE
    	bli_cntx_init_zen3.c
		bli_family_zen3.h
   	)

target_include_directories(${TARGET_NAME}
    PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}
	)