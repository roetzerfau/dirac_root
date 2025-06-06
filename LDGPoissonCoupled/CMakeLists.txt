##
#  CMake script for the LDGPoissonCoupled program:
##

# Set the name of the project and target:
SET(TARGET "LDGPoissonCoupled")

# Declare all source files the target consists of:
SET(TARGET_SRC
  ${TARGET}.cc
  Functions.cc
  )

# Define the output that should be cleaned:
SET(CLEAN_UP_FILES *.vtu *.pvtu *.visit)

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 3.3.0)
#set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
FIND_PACKAGE(deal.II 9.5.1
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()
#DEAL_II_WITH_TRILINOS_TPETRA=ON
#
# Are all dependencies fullfilled?
#
IF( NOT DEAL_II_WITH_MPI OR
    NOT DEAL_II_WITH_P4EST OR
    NOT DEAL_II_WITH_TRILINOS )
  MESSAGE(FATAL_ERROR "
Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_MPI = ON
    DEAL_II_WITH_P4EST = ON
    DEAL_II_WITH_TRILINOS = ON
One or all of these are OFF in your installation but are required for this tutorial step."
    )
ENDIF()

#DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})

DEAL_II_INVOKE_AUTOPILOT()
