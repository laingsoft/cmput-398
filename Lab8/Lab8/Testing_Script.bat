@echo off

set TEST_DIR=Dataset\Inclusive\Test\
set APP=Submission\Inclusive.exe
set NAME=Inclusive Scan

DEL /S %TEST_DIR%*myOutput.raw

echo var timestamp = '%time%'; > Marks.js

echo var text = '{"Marks": [' + >> Marks.js
echo '{"Section": "%NAME%", "Tests": [' +   >> Marks.js

FOR /L %%x IN (0,1,11) DO (
echo Testing %NAME% with test %%x
echo '{"Test": "Test %%x", "Output": [' +  >> Marks.js
set TEST=%TEST_DIR%%%x\
call:runTest
IF %%x LSS 11 (
	echo ']},' +  >> Marks.js
) ELSE (
	echo ']}' +  >> Marks.js
)
)

echo ']},' +  >> Marks.js

set TEST_DIR=Dataset\Exclusive\Test\
set APP=Submission\Exclusive.exe
set NAME=Exclusive Scan

DEL /S %TEST_DIR%*myOutput.raw

echo '{"Section": "%NAME%", "Tests": [' +   >> Marks.js

FOR /L %%x IN (0,1,11) DO (
echo Testing %NAME% with test %%x
echo '{"Test": "Test %%x", "Output": [' +  >> Marks.js
set TEST=%TEST_DIR%%%x\
call:runTest
IF %%x LSS 11 (
	echo ']},' +  >> Marks.js
) ELSE (
	echo ']}' +  >> Marks.js
)
)

echo ']}' +  >> Marks.js
echo ']}'; >> Marks.js

echo.&goto:eof

:runTest
%APP% -e %TEST%output.raw -i %TEST%input.raw -o %TEST%myOutput.raw -t vector > tmp.txt
for /f "tokens=*" %%a in (tmp.txt) do (
	echo '%%a,' + >> Marks.js
)

echo '{"data": {"Done": true, "message": "The test is done"}, "type": "Done"}' + >> Marks.js

DEL tmp.txt

goto:eof