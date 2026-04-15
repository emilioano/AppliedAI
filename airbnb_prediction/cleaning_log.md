## Datarening – prisvariabel

Variabeln `price` var i originaldatan lagrad som text och innehöll valutasymboler (`$`) samt tusentalsavgränsare (`,`). 

För att kunna använda variabeln i analys och modellering rensades dessa tecken bort och variabeln konverterades till numeriskt format (float).

Detta möjliggjorde vidare statistisk analys samt användning i regressionsmodeller.


## Datarening – extrema prisvärden

Endast de högsta prisvärdena exkluderades (över 99:e percentilen), då dessa sannolikt representerar outliers.

De lägsta prisvärdena behölls, eftersom det inte finns något som tyder på att dessa är felaktiga, utan snarare speglar variation i standard och storlek.

## Datarening – uthyrningstyp

Observationer med uthyrningstypen "Hotel room" exkluderades eftersom det bara finns 1 sån och airbnb inte hyr ut hotellrum.

## Datatransformation – property_group

Variabeln `property_type` innehöll ett stort antal kategorier med varierande detaljnivå, där både boendetyp och uthyrningsform blandades.

För att möjliggöra jämförbar analys skapades en förenklad variabel, `property_group`, där boenden grupperades i tre kategorier:

- apartment (lägenheter och liknande)
- house (hus och fristående boenden)
- unique (ovanliga boendetyper, t.ex. båt eller husbil)

Denna uppdelning gör det möjligt att analysera effekten av boendetyp separat från uthyrningstyp (`room_type`).

## Datarening – shared rooms

En manuell granskning av annonser med `room_type = "Shared room"` visade att dessa i praktiken avser en bädd i sovsal eller delat rum, snarare än en privat boendeyta.

Dessa observationer exkluderades därför från modellen, eftersom priset inte är direkt jämförbart med priset för privata rum eller hela bostäder.