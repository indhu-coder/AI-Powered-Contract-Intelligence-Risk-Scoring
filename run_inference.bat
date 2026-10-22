```bat
@echo off
cd /d D:\ContractIntelligence

call contractenv\Scripts\activate.bat

python test_inference.py

pause
```