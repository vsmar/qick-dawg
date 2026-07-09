The reoptimizer hook relies on a file in the qdlutils suite.

This enables reoptimization via the niDAQ based on the same optimization logic in QDL scan.
Overhauls of scan logic will likely render automatic reoptimization no longer useful, 
but until then it is possible to run experiments with automatic reoptimization.

This folder contains a backup copy of the original script in case it goes missing. 

As of 7/8/2026 the script lives here, and this is where the reopt hook expects it:
"C:\Users\QT3 User Facility\qdl-utils\src\qdlutils\applications\qdlscan\reoptimizer.py"

Disclaimer:
    Reoptimization script and hooks were vibecoded and only checked for functionality.