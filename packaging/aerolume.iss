; Inno Setup script for Aerolume (DESIGN.md section 12). Compiled by packaging/build.py:
;
;   ISCC /DAppVersion=0.1.0 /DVariant=cuda /DSourceDir=<dist\Aerolume> /DOutputDir=<out>
;        /DIconFile=<gen\aerolume.ico> packaging\aerolume.iss
;
; Design choices:
; * Per-user install (no administrator prompt) into %LOCALAPPDATA%\Programs\Aerolume, like
;   VS Code's user installer; "/ALLUSERS" on the command line installs for everyone instead.
; * The whole _internal folder is deleted before files are copied, so an upgrade never leaves a
;   stale DLL from an older torch or Qt next to the new ones (Windows would load either).
; * The project association is for files with the .aerolume extension: a project is a folder
;   named "<name>.aerolume" holding project.json and a "project.aerolume" marker file, and
;   double-clicking the marker opens the folder (aerolume.app.platform.resolve_open_target).
; * AppMutex makes setup and the uninstaller wait until Aerolume is closed (the app holds the
;   mutex: aerolume.app.platform.hold_app_mutex). Shortcuts carry the app's AppUserModelID so a
;   pinned shortcut and the running window share one taskbar button.
; * User data (%APPDATA%\Aerolume: settings, recent projects, logs) and projects are never
;   touched by the uninstaller.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef VersionQuad
  #define VersionQuad "0.0.0.0"
#endif
#ifndef Variant
  #define Variant "cuda"
#endif
#ifndef SourceDir
  #error SourceDir (the PyInstaller dist\Aerolume folder) must be defined: /DSourceDir=...
#endif
#ifndef OutputDir
  #define OutputDir "."
#endif
#ifndef IconFile
  #error IconFile (the generated aerolume.ico) must be defined: /DIconFile=...
#endif
#ifndef Compression
  #define Compression "lzma2/max"
#endif
#ifndef LzmaThreads
  #define LzmaThreads "4"
#endif

#define AppName "Aerolume"
#define AppExe "Aerolume.exe"
#define AppId "{6B0C2E1F-6A0B-4C55-9C0E-2B7F2E9A4D31}"
#define AppUserModelId "Aerolume.Aerolume.Desktop"
#define AppMutexName "Aerolume.Aerolume.Desktop.Running"
#define ProgId "Aerolume.Project"
#if Variant == "cuda"
  #define VariantLabel "NVIDIA CUDA"
#else
  #define VariantLabel "CPU"
#endif

[Setup]
AppId={{#AppId}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion} ({#VariantLabel})
AppPublisher=Aerolume contributors
AppCopyright=Copyright Aerolume contributors. Apache License 2.0.
AppComments=Open-source 2D and 3D aerodynamics studio
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableWelcomePage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline
UsedUserAreasWarning=no
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Windows 10 1809 or later (the CUDA 13 runtime and Qt 6 both require Windows 10).
MinVersion=10.0.17763
OutputDir={#OutputDir}
OutputBaseFilename=Aerolume-{#AppVersion}-{#Variant}-setup
; Tab-separated list of every file compressed into setup (source path, size, SHA-256).
; packaging/build.py checks it against the app folder, which verifies the installer's contents
; without running it (buildlib.verify_installer_manifest).
OutputManifestFile=Aerolume-{#AppVersion}-{#Variant}-manifest.txt
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName} {#AppVersion} ({#VariantLabel})
LicenseFile={#SourceDir}\LICENSE
Compression={#Compression}
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads={#LzmaThreads}
WizardStyle=modern dynamic
WizardSizePercent=110
ChangesAssociations=yes
AppMutex={#AppMutexName}
CloseApplications=yes
RestartApplications=no
SetupLogging=yes
VersionInfoVersion={#VersionQuad}
VersionInfoProductName={#AppName}
VersionInfoProductVersion={#AppVersion}
VersionInfoProductTextVersion={#AppVersion} ({#VariantLabel})
VersionInfoDescription={#AppName} {#AppVersion} setup ({#VariantLabel})
VersionInfoCompany=Aerolume contributors
VersionInfoCopyright=Apache License 2.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
english.AssocTask=Open Aerolume project files (.aerolume) with Aerolume
english.AssocGroup=File associations:
english.ProjectFileType=Aerolume project
english.NoNvidiaDriver=No NVIDIA graphics driver was found on this computer.%n%nThis build includes the NVIDIA CUDA libraries, but without a driver Aerolume runs every simulation on the CPU, which is much slower. Install the current NVIDIA driver to use the GPU.%n%nContinue with the installation?

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "associate"; Description: "{cm:AssocTask}"; GroupDescription: "{cm:AssocGroup}"

[InstallDelete]
Type: filesandordirs; Name: "{app}\_internal"
Type: filesandordirs; Name: "{app}\docs"
Type: filesandordirs; Name: "{app}\validation"
Type: filesandordirs; Name: "{app}\licenses"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{userdocs}"; AppUserModelID: "{#AppUserModelId}"; Comment: "Open-source 2D and 3D aerodynamics studio"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{userdocs}"; AppUserModelID: "{#AppUserModelId}"; Comment: "Open-source 2D and 3D aerodynamics studio"; Tasks: desktopicon

[Registry]
; ProgID and extension (HKA = HKCU for a per-user install, HKLM for /ALLUSERS)
Root: HKA; Subkey: "Software\Classes\.aerolume"; ValueType: string; ValueName: ""; ValueData: "{#ProgId}"; Flags: uninsdeletevalue; Tasks: associate
Root: HKA; Subkey: "Software\Classes\.aerolume\OpenWithProgids"; ValueType: string; ValueName: "{#ProgId}"; ValueData: ""; Flags: uninsdeletevalue; Tasks: associate
Root: HKA; Subkey: "Software\Classes\{#ProgId}"; ValueType: string; ValueName: ""; ValueData: "{cm:ProjectFileType}"; Flags: uninsdeletekey; Tasks: associate
Root: HKA; Subkey: "Software\Classes\{#ProgId}"; ValueType: string; ValueName: "AppUserModelID"; ValueData: "{#AppUserModelId}"; Tasks: associate
Root: HKA; Subkey: "Software\Classes\{#ProgId}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"",0"; Tasks: associate
Root: HKA; Subkey: "Software\Classes\{#ProgId}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: associate
; "Open with" list entry and the application's friendly name
Root: HKA; Subkey: "Software\Classes\Applications\{#AppExe}"; ValueType: string; ValueName: "FriendlyAppName"; ValueData: "{#AppName}"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\Applications\{#AppExe}\SupportedTypes"; ValueType: string; ValueName: ".aerolume"; ValueData: ""
Root: HKA; Subkey: "Software\Classes\Applications\{#AppExe}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""
; "App Paths" so Win+R "aerolume" and "start aerolume" find the program
Root: HKA; Subkey: "Software\Microsoft\Windows\CurrentVersion\App Paths\{#AppExe}"; ValueType: string; ValueName: ""; ValueData: "{app}\{#AppExe}"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Microsoft\Windows\CurrentVersion\App Paths\{#AppExe}"; ValueType: string; ValueName: "Path"; ValueData: "{app}"

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Created at run time by user scripts despite sys.dont_write_bytecode in older builds.
Type: filesandordirs; Name: "{app}\validation\__pycache__"

[Code]
#if Variant == "cuda"
function InitializeSetup(): Boolean;
begin
  Result := True;
  { nvcuda.dll is installed by every NVIDIA display driver; without it CUDA cannot start. }
  if not FileExists(ExpandConstant('{sys}\nvcuda.dll')) then
    Result := SuppressibleMsgBox(CustomMessage('NoNvidiaDriver'), mbConfirmation, MB_YESNO, IDYES) = IDYES;
end;
#endif
