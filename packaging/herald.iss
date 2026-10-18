; Установщик Herald для Windows (Inno Setup 6, по образцу Scribe 2.4).
; Внутри: программа, ffmpeg (LGPL). Моделей и образцов голосов внутри нет:
; модели качаются при первом запуске, голос пользователь добавляет сам.
#ifndef AppVersion
  #define AppVersion "3.8"
#endif

[Setup]
AppId={{3B1F6C8E-6E0B-4D2A-9C5E-7A41D2B9F0A3}
AppName=Herald
AppVersion={#AppVersion}
AppVerName=Herald {#AppVersion}
AppPublisher=Herald
DefaultDirName={autopf}\Herald
DefaultGroupName=Herald
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist
OutputBaseFilename=Herald-Setup-{#AppVersion}
SetupIconFile=herald.ico
UninstallDisplayIcon={app}\Herald.exe
Compression=lzma2/normal
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\Herald\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Herald"; Filename: "{app}\Herald.exe"
Name: "{group}\Удалить Herald"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Herald"; Filename: "{app}\Herald.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Herald.exe"; Description: "{cm:LaunchProgram,Herald}"; Flags: nowait postinstall skipifsilent
