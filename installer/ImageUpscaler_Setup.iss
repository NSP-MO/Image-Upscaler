; Inno Setup 6 Script for Image Upscaler
; Self-Contained Zero-Dependency Windows Desktop Installer

#define MyAppName "Image Upscaler"
#define MyAppVersion "0.2.0"
#define MyAppPublisher "NSP-MO"
#define MyAppURL "https://github.com/NSP-MO/Image-Upscaler"
#define MyAppExeName "ImageUpscaler.exe"

[Setup]
; App Metadata
AppId={{4F22C0A5-7D39-4E92-821B-47391F0992AB}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} v{#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}

; Destination Directories
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes

; Output Configuration (relative to script location in installer/)
OutputDir=output
OutputBaseFilename=ImageUpscaler_Setup_v0.2.0_x64
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern

; Architecture & Platform Requirements
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; Uninstaller configuration
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Include all published self-contained files from publish directory
Source: "..\publish\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
Name: "{app}\weights"

[Icons]
Name: "{autoprograms}\{#MyAppName}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autoprograms}\{#MyAppName}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\weights"
Type: filesandordirs; Name: "{app}"
