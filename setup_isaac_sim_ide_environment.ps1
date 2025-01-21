$scriptName = "setup_isaac_sim_ide_environment.ps1"
$directoryName = "C:\Temp"
$downloadUrl = "https://raw.githubusercontent.com/Auromix/auro_sim/main/scripts"
$fullScriptPath = "$directoryName\$scriptName"

Invoke-WebRequest -Uri "$downloadUrl/$scriptName" -OutFile $fullScriptPath
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& $fullScriptPath
Remove-Item $fullScriptPath