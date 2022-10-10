### Put hooks from tracked /.githooks into .git/hooks
# First need to find the top-level directory of the git repository
execute_process(
  COMMAND ${GIT_EXECUTABLE} rev-parse --show-toplevel
  OUTPUT_VARIABLE GIT_TOP_LEVEL
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
# N.B. The variable comes back from 'git describe' with a line feed on the end, so we need to lose that
string(REGEX MATCH "(.*)[^\n]" GIT_TOP_LEVEL ${GIT_TOP_LEVEL})
# Prefer symlinks on platforms that support it so we don't rely on cmake running to be up-to-date On Windows, we
# have to copy the file
if(WIN32)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${GIT_TOP_LEVEL}/.githooks/commit-msg ${GIT_TOP_LEVEL}/.git/hooks
  )
else()
  execute_process(
   COMMAND ${CMAKE_COMMAND} -E create_symlink ${GIT_TOP_LEVEL}/.githooks/commit-msg
             ${GIT_TOP_LEVEL}/.git/hooks/commit-msg
  )
endif()

### Set up pre-commit, adds script to .git/hooks
# Windows should use downloaded ThirdParty version of pre-commit.cmd Everybody else should find one in their PATH
find_program(
  PRE_COMMIT_EXE
  NAMES pre-commit
  HINTS ~/.local/bin/ "${MSVC_PYTHON_EXECUTABLE_DIR}/Scripts/"
)
if(NOT PRE_COMMIT_EXE)
  message(FATAL_ERROR "Failed to find pre-commit ensure conda environment is active")
endif()
if(WIN32)
  execute_process(
    COMMAND "${PRE_COMMIT_EXE}" install --overwrite
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    RESULT_VARIABLE PRE_COMMIT_RESULT
  )
  if(NOT PRE_COMMIT_RESULT EQUAL "0")
    message(FATAL_ERROR "Pre-commit install failed with ${PRE_COMMIT_RESULT}")
  endif()
  # Create pre-commit script wrapper to use mantid third party python for pre-commit
  file(TO_CMAKE_PATH $ENV{CONDA_PREFIX} CONDA_SHELL_PATH)
  file(RENAME "${PROJECT_SOURCE_DIR}/.git/hooks/pre-commit" "${PROJECT_SOURCE_DIR}/.git/hooks/pre-commit-script.py")
  file(
    WRITE "${PROJECT_SOURCE_DIR}/.git/hooks/pre-commit"
    "#!/usr/bin/env sh\n${CONDA_SHELL_PATH}/Scripts/wrappers/conda/python.bat ${PROJECT_SOURCE_DIR}/.git/hooks/pre-commit-script.py"
  )
else() # linux as osx
  execute_process(
    COMMAND bash -c "${PRE_COMMIT_EXE} install"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    RESULT_VARIABLE STATUS
  )
  if(STATUS AND NOT STATUS EQUAL 0)
    message(
      FATAL_ERROR
        "Pre-commit tried to install itself into your repository, but failed to do so. Is it installed on your system?"
    )
  endif()
  endif()