# Disease and pest descriptions (sources for `DISEASE_INFO`)

The chatbot answers "What is ...?" questions with the text below (the `DISEASE_INFO` table in
`agriassist.py`). There is one entry for each of the model's class names; "Leaf blight" and
"Healthy" exist under both crops, so 12 model classes give 10 entries.

**Why the wording is careful.** The CCMT dataset paper (Mensah et al., 2023) names the classes
but does not describe any of them, and does not say which pathogen is behind the general
labels "leaf blight", "leaf spot" and "leaf curl". So those entries describe the class in
general terms, give the best-known example, and tell the farmer to confirm with an extension
officer. None of the entries recommends a chemical: treatment decisions stay with the farmer
and the spraying map, as the safety rules require.

| # | Class | Crop | Type |
|---|---|---|---|
| 1 | Fall armyworm | maize | pest |
| 2 | Grasshopper | maize | pest |
| 3 | Leaf beetle | maize | pest |
| 4 | Leaf blight | maize and tomato | disease |
| 5 | Leaf spot | maize | disease |
| 6 | Streak virus | maize | disease (virus) |
| 7 | Septoria leaf spot | tomato | disease |
| 8 | Verticillium wilt | tomato | disease |
| 9 | Leaf curl | tomato | disease (virus) |
| 10 | Healthy | maize and tomato | baseline |

---

### 1. Fall armyworm (maize, pest)
Fall armyworm is a caterpillar that feeds on young maize. Small caterpillars scrape the leaf
surface and leave thin, papery "windows"; bigger ones chew ragged holes, hide in the whorl and
can bore into the cob.

Sources: Pacific Pests, Pathogens & Weeds fact sheet 401, *Fall armyworm*
([lucidcentral.org](https://apps.lucidcentral.org/ppp/text/web_full/entities/fall_armyworm_401.htm));
icipe, *Fall armyworm factsheet*
([technologytransfer.icipe.org](https://technologytransfer.icipe.org/wp-content/uploads/2021/04/A3-Fall-armyworm-Factsheet.pdf)).

### 2. Grasshopper (maize, pest)
Grasshoppers chew ragged holes in maize leaves; in bad cases only the midrib is left, and they
can also clip the silks. Damage usually starts at field edges next to grass or weeds and is
worse in dry years.

Sources: Iowa State University Extension, *Hungry hungry (grass) hoppers* (2024)
([crops.extension.iastate.edu](https://crops.extension.iastate.edu/cropnews/2024/07/hungry-hungry-grass-hoppers));
Purdue Extension, *Grasshoppers (Corn)*
([ag.purdue.edu](https://ag.purdue.edu/department/entm/extension/field-crops-ipm/corn/grasshoppers.html)).

### 3. Leaf beetle (maize, pest)
Leaf beetles are small beetles whose adults chew and shred maize leaves, often in patches near
field edges. In some species the larvae live in the soil and feed on roots, which can kill
young plants.

Source: Queensland Government, *Leaf beetles in field crops*
([business.qld.gov.au](https://www.business.qld.gov.au/industries/farms-fishing-forestry/agriculture/biosecurity/plants/insects/field-crop/leaf-beetles)).
The dataset does not name the beetle species.

### 4. Leaf blight (maize and tomato, disease)
**On maize:** long grey-green to tan lesions on the leaves, usually starting on the lower
leaves. Northern corn leaf blight is the best-known example; its fungus survives on old maize
residue left in the field.
**On tomato:** dark spots, often with rings like a target and a yellow edge, starting on the
older, lower leaves. Early blight is the best-known example, and it is favoured by warm, wet
weather.
The dataset does not say which blight it is, so ask an extension officer to confirm.

Sources: Crop Protection Network, *Northern Corn Leaf Blight of Corn*
([cropprotectionnetwork.org](https://cropprotectionnetwork.org/encyclopedia/northern-corn-leaf-blight-of-corn));
University of Minnesota Extension, tomato diagnostic pages for early blight
([apps.extension.umn.edu](https://apps.extension.umn.edu/garden/diagnose/plant/vegetable/tomato/stemsspots.html)).

### 5. Leaf spot (maize, disease)
Leaf spot is a general name for fungal diseases that leave spots or lesions on maize leaves.
Gray leaf spot, for example, shows as narrow, rectangular, tan-to-grey lesions between the leaf
veins, starting on the lower leaves. The dataset does not say which fungus, so ask an extension
officer to confirm.

Source: Crop Protection Network, *Gray Leaf Spot of Corn*
([cropprotectionnetwork.org](https://cropprotectionnetwork.org/encyclopedia/gray-leaf-spot-of-corn)).

### 6. Streak virus (maize, disease)
Maize streak virus is spread by small leafhoppers. Leaves get narrow white-to-yellow streaks
along the veins, and early infection can stunt the plant and spoil the cobs. There is no
chemical cure for the virus, so resistant varieties and keeping leafhopper numbers down are the
main defences.

Sources: Plantix plant-disease library, *Maize streak virus*
([plantix.net](https://www.plantix.net/en/library/plant-diseases/200023/maize-leaf-streak-virus));
IITA / CIMMYT, *Development and regional deployment of streak virus resistant maize germplasm:
an overview* ([biblio1.iita.org](https://biblio1.iita.org/handle/20.500.12478/4306)).

### 7. Septoria leaf spot (tomato, disease)
Septoria leaf spot is a fungal disease that makes many small, round spots with a dark edge and
a grey or tan centre, starting on the lowest leaves. It spreads in warm, humid weather when
water splashes spores up from the soil.

Source: University of Minnesota Extension, *Tomato leaf spot diseases*
([extension.umn.edu](https://extension.umn.edu/plant-diseases/tomato-leaf-spot-diseases)).

### 8. Verticillium wilt (tomato, disease)
Verticillium wilt is a fungal disease that lives in the soil. Leaves turn yellow, often in a
wedge shape, and wilt, starting with the lower leaves and often on one side of the plant;
cutting the stem near the soil shows tan streaks inside.

Source: University of Minnesota Extension, *Tomato: wilted leaves* diagnostic page
([apps.extension.umn.edu](https://apps.extension.umn.edu/garden/diagnose/plant/vegetable/tomato/leaveswilted.html)).

### 9. Leaf curl (tomato, disease)
Leaf curl in tomato is best known from tomato yellow leaf curl virus, which whiteflies carry:
young leaves turn yellow and curl, plants become stunted and bushy, and flowers drop before
fruit sets. Nutrient problems and herbicide damage can look similar, so ask an extension
officer to confirm.

Sources: NC State Extension, *Tomato yellow leaf curl virus*
([content.ces.ncsu.edu](https://content.ces.ncsu.edu/tomato-yellow-leaf-curl-virus));
NSW Department of Primary Industries, *Tomato yellow leaf curl virus in Australia*, Primefact 220.

### 10. Healthy (maize and tomato, baseline)
The system found no sign of the pests or diseases it checks for on this leaf. That is not a
guarantee, so keep scouting and look closely at any plant that seems different.

Source: the dataset's "healthy" class (Mensah et al., 2023, below).

---

**Dataset reference.** Mensah, P. K., Akoto-Adjepong, V., Adu, K., et al. (2023). CCMT: Dataset
for crop pest and disease detection. *Data in Brief*, 49, 109306.
<https://doi.org/10.1016/j.dib.2023.109306>
