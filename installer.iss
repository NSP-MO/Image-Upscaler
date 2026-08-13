[Setup]
AppId={{4F22C0A5-7D39-4E92-821B-47391F0992AB}
AppName=Image Upscaler
AppVersion=0.2.0
AppPublisher=NSP-MO
AppPublisherURL=https://github.com/NSP-MO/Image-Upscaler
DefaultDirName={userlocalappdata}\Programs\Image Upscaler
DefaultGroupName=Image Upscaler
UninstallDisplayIcon={app}\ImageUpscaler.exe
Compression=lzma2/ultra64
SolidCompression=yes
OutputDir=.
OutputBaseFilename=ImageUpscaler-Setup-win-x64
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Types]
Name: "full"; Description: "Full Installation (Image Upscaler + Python & PyTorch Neural Engine Setup)"
Name: "compact"; Description: "Compact Installation (Image Upscaler App only - High-Fidelity Native C# Engine)"
Name: "custom"; Description: "Custom Installation"; Flags: iscustom

[Components]
Name: "core"; Description: "Image Upscaler Core Application (WPF Desktop App & Native Engine)"; Types: full compact custom; Flags: fixed
Name: "python"; Description: "Python Runtime & PyTorch Neural Dependencies (Required for Real-ESRGAN, SwinIR, DAT Neural Models)"; Types: full

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "publish\ImageUpscaler.exe"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "publish\requirements.txt"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "publish\models\*"; DestDir: "{app}\models"; Components: core; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
Name: "{app}\weights"

[UninstallDelete]
Type: filesandordirs; Name: "{app}\weights"
Type: filesandordirs; Name: "{app}\models"
Type: filesandordirs; Name: "{app}"

[Icons]
Name: "{group}\Image Upscaler"; Filename: "{app}\ImageUpscaler.exe"
Name: "{autodesktop}\Image Upscaler"; Filename: "{app}\ImageUpscaler.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\ImageUpscaler.exe"; Description: "{cm:LaunchProgram,Image Upscaler}"; Flags: nowait postinstall skipifsilent
