# Manus – Airbnb-priser i Stockholm

---

## SLIDE 2 – Inledning & syfte
• Bygga en regressionsmodell  
• Förstå vad som påverkar pris  
• Arbeta med verklig och stökig data  

(Säg:)
Vi analyserar Airbnb-priser i Stockholm med målet att både bygga en modell och förstå vad som driver pris.

---

## SLIDE 3 – Dataset och första observation
• Airbnb-listningar i Stockholm  
• Variabler: pris, område, boendetyp, kapacitet  
• visuell analys med pairplot  

(Visa:)
Pairplot + graf över pris per område

(Säg:)
Vi började med att utforska datan visuellt.

Då såg vi direkt stora variationer i pris,
och att vissa förorter verkade lika dyra som innerstaden.

Det blev vår första indikation på att något i datan inte stämmer.

---

---

## SLIDE 4 – Problem 1: Prisvariabeln
• price var lagrad som text  
• innehöll $ och ,  
• kunde inte användas direkt  

(Visa:)
Råa price-värden

(Kod:)


(Säg:)
Vi började med att göra priset numeriskt.

---

## SLIDE 5 – Problem 2: Outliers
• mycket höga prisvärden drog snett  
• vi tog bort endast de högsta  
• de lägsta behölls  

(Visa:)
Histogram eller boxplot före/efter

(Kod:)
upper = df["price"].quantile(0.99)
df = df[df["price"] <= upper]

(Säg:)
Vi tog bort extrema toppvärden men behöll variation i lägre priser.

---


---

## SLIDE 6 – problem3: storlek 
• beds, bedrooms och accommodates hänger inte ihop  
• exempel: 1 sovrum → 12 gäster  
• vissa boenden har 0 sängar  

• vi fyllde saknade värden  
• justerade extrema relationer  
• behöll datan istället för att ta bort  

(Visa:)
Scatterplot (beds vs accommodates) + ev före/efter

(Säg:)
Här såg vi att datan inte var logiskt konsekvent.


---

## SLIDE 8 – Problem4: shared rooms
• manuellt granskade annonser  
• pris gäller en bädd i sovsal  
• accommodates och beds beskriver hela rummet  
• leder till skev storlek och prisrelation  

(Visa:)
Exempel-listningar

(Kod:)
df.loc[df["room_type"] == "Shared room", ["beds", "accommodates"]] = 1


Vi korrigerade detta så varje rad motsvarar en faktisk bädd.

---

## SLIDE 9 – Datatransformation: property_group
• property_type var för detaljerad  
• blandade boendetyp och uthyrningsform  (visa bild)
• behövde förenklas  

(Kod:)


(Säg:)
Vi förenklade property_type till fyra tydliga grupper för att kunna analysera boendetyp separat.


## SLIDE 10 – Bevis: varför förorter såg dyra ut
• när vi bryter ut boendetyp förändras bilden  
• hus driver upp pris i vissa områden  
• lägenheter ligger lägre  

(Visa:)
Boxplot med property_group

(Säg:)
Nu ser vi att det inte bara är läget – utan även typen av boenden.

---

## SLIDE 11 – Modell & transformation
• pris är snedfördelat syns med shapiro
• log-transform används = ändå inte normalfördelat men tillräckligt bra 

(Kod:)
cleandf["log_price"] = np.log1p(cleandf["price"])

• linjär regression  
• features: storlek, område, boendetyp  

(Säg:)
Vi transformerar priset och bygger en regressions modell på den rensade datan.

---

## SLIDE 12 – Träning vs test
• Gör en train/test split (80/20)  
• jämför modellens prestation  

(Visa:)
R² träning vs test + MAE 


(Säg:)
Det viktiga är att modellen fungerar lika bra på ny data som på träningsdata.

Om resultaten är lika → modellen generaliserar bra.

---

## SLIDE 13 – Resultat
• stabil modell  
• rimliga prediktioner  
• förbättras av datarensning  

(Visa:)
Predicted vs actual (graf)

(Säg:)
Efter datarensningen får vi en modell som ger rimliga prediktioner.

---

## SLIDE 14 – Användning & slutsats
• hitta överprisade boenden  
• identifiera bra deals  
• datakvalitet avgör resultat  

(Visa:)
Exempel från priskalkylator

(Säg:)
Modellen kan användas för att bedöma om ett pris är rimligt.

Den viktigaste slutsatsen är att datakvalitet avgör modellens kvalitet.