; Windows installer for the desktop app (Inno Setup 6). Built by .github/workflows/desktop.yml:
;   iscc /DAppVersion=0.2.0 packaging\windows.iss
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{0C11A234-F205-44C3-A828-C997DBD24FB9}
AppName=plainquant
AppVersion={#AppVersion}
AppPublisher=plainquant contributors
AppPublisherURL=https://github.com/Nikshay1/plainquant
DefaultDirName={localappdata}\Programs\plainquant
DefaultGroupName=plainquant
DisableProgramGroupPage=yes
; Installs for the current user, so no admin prompt.
PrivilegesRequired=lowest
OutputDir=dist
OutputBaseFilename=plainquant-windows-setup
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\plainquant.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "dist\plainquant\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{autoprograms}\plainquant"; Filename: "{app}\plainquant.exe"
Name: "{autodesktop}\plainquant"; Filename: "{app}\plainquant.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\plainquant.exe"; Description: "Open plainquant"; Flags: nowait postinstall skipifsilent
