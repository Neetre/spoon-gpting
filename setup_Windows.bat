@ECHO OFF

REM SETUP THE ENVIRONMENT

REM Create a Python virtual environment if it doesn't already exist
IF NOT EXIST ".venv" (
    call python -m venv .venv
    IF ERRORLEVEL 1 (
        ECHO Failed to create virtual environment
        EXIT /B 1
    )
)

REM Activate the virtual environment
call .venv/Scripts/activate
IF ERRORLEVEL 1 (
    ECHO Failed to activate virtual environment
    EXIT /B 1
)

REM Install required Python packages
call pip install -r requirements.txt
IF ERRORLEVEL 1 (
    ECHO Failed to install required packages
    EXIT /B 1
)