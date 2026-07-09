To get this back up working with aliases in PowerShell, update your Windows PowerShell Profile.

For the VQLM this is:
C:\Users\QT3 User Facility\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1


Add the following lines:
function Invoke-LaserOn  { & "C:\Users\QT3 User Facility\Documents\qick-dawg\vqlm_scripts\Invoke-Laser.ps1" -State On }
function Invoke-LaserOff { & "C:\Users\QT3 User Facility\Documents\qick-dawg\vqlm_scripts\Invoke-Laser.ps1" -State Off }

Set-Alias laser-on  Invoke-LaserOn
Set-Alias laser-off Invoke-LaserOff


More robust fix (currently implemented): 
To avoid issues with branching in this repo, the files in this folder have been copied to:
C:\Users\QT3 User Facility\RFSoC

But with the updated paths for laser_control.py inside Invoke-Laser.ps1, and the ps1 alias is then set-up as:
function Invoke-LaserOn  { & "C:\Users\QT3 User Facility\RFSoC\Invoke-Laser.ps1" -State On }
function Invoke-LaserOff { & "C:\Users\QT3 User Facility\RFSoC\Invoke-Laser.ps1" -State Off }

Set-Alias laser-on  Invoke-LaserOn
Set-Alias laser-off Invoke-LaserOff