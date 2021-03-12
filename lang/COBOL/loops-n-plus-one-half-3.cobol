IDENTIFICATION DIVISION.
PROGRAM-ID. LOOP-1p5-NOADV-GOTO.
DATA DIVISION.
WORKING-STORAGE SECTION.
01  I	PIC 99	VALUE 1.
	88	END-LIST	VALUE 10.
01	I-OUT	PIC Z9.
PROCEDURE DIVISION.
01-LOOP.
	MOVE I TO I-OUT.
	DISPLAY FUNCTION TRIM(I-OUT) WITH NO ADVANCING.
	IF END-LIST GO TO 02-DONE.
	DISPLAY ", " WITH NO ADVANCING.
	ADD 1 TO I.
	GO TO 01-LOOP.
02-DONE.
	STOP RUN.
	END-PROGRAM.