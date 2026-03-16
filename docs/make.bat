@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation (Windows)
REM
REM Usage:
REM   make.bat html    — build HTML documentation
REM   make.bat clean   — remove build artefacts

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found.
	echo.Install Sphinx with:
	echo.    pip install sphinx
	exit /b 1
)

if "%1" == ""      goto help
if "%1" == "help"  goto help
if "%1" == "clean" goto clean
if "%1" == "html"  goto html

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:clean
%SPHINXBUILD% -M clean %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:html
%SPHINXBUILD% -b html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
if errorlevel 1 exit /b 1
echo.
echo.Build finished. Open %BUILDDIR%\index.html in your browser.
goto end

:end
popd
