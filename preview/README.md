# Image Sample Gallery

A small, downscaled sample of the image seed data — 11 categories,
222,931 images in total on
[HuggingFace](https://huggingface.co/datasets/mlech26l/liquidrandom-data).

These previews are 384px JPEGs so the repository stays small. The
real images are ~1K WebP at their native aspect ratio; fetch them with
`liquidrandom.image(category, tags)`. Every image below is reachable from the
package — the filenames carry the `chain_id` it belongs to.

```python
import liquidrandom

img = liquidrandom.indoor_scene(tags=["no_people"])
chain = liquidrandom.image_chain_of(img)   # the base image and its edits
```

Regenerate this page with `python seed_generation/make_preview_gallery.py`.

## Indoor Scenes

`liquidrandom.indoor_scene()` — 20,158 images.
Tags: `people`, `no_people`, `setting:residential`, `setting:commercial`, `setting:institutional`, `lighting:natural`, `lighting:artificial`, `lighting:dim`, `tidiness:tidy`, `tidiness:cluttered`

| | | | |
|---|---|---|---|
| <img src="images/indoor_scene/base_b345876a3e46e461.jpg" width="200"> | <img src="images/indoor_scene/base_9d7a07ca9871acd5.jpg" width="200"> | <img src="images/indoor_scene/base_b08355f1bf3a26a1.jpg" width="200"> | <img src="images/indoor_scene/base_f80e5a3b25e5a69e.jpg" width="200"> |
| Medium shot of an office cluster with green desks and binder stacks.<br>`people` `setting:commercial` `lighting:natural` `tidiness:cluttered` | A view from the entrance doors shows guests lining up on a red carpet.<br>`people` `setting:commercial` `lighting:artificial` `tidiness:tidy` | A low angle shot focusing on puddle reflections in a sunlit warehouse atrium with a collapsed catwalk.<br>`no_people` `setting:commercial` `lighting:natural` `tidiness:cluttered` | A tight crop of a 1970s workshop pegboard wall with flash lighting highlighting wood offcuts.<br>`no_people` `setting:residential` `lighting:artificial` `tidiness:cluttered` |

An edit chain — one base image and three edits, each applied to the base or to
an earlier turn:

| | | | |
|---|---|---|---|
| <img src="images/indoor_scene/chain_0cab0dcafe0f8b71_t0.jpg" width="200"> | <img src="images/indoor_scene/chain_0cab0dcafe0f8b71_t1.jpg" width="200"> | <img src="images/indoor_scene/chain_0cab0dcafe0f8b71_t2.jpg" width="200"> | <img src="images/indoor_scene/chain_0cab0dcafe0f8b71_t3.jpg" width="200"> |
| **base image** | *edit 1 of turn 0*<br>Change the wood tones from warm pine to cool grey-washed wood. | *edit 2 of turn 1*<br>Make the space significantly messier with scattered books and cups. | *edit 3 of turn 0*<br>Add a sleeping dog on the rug near the hearth. |

## Outdoor Scenes

`liquidrandom.outdoor_scene()` — 20,371 images.
Tags: `people`, `no_people`, `setting:urban`, `setting:rural`, `setting:nature`, `time:day`, `time:night`, `time:dawn_dusk`, `weather:clear`, `weather:overcast`, `weather:rain`, `weather:snow`, `weather:fog`

| | | | |
|---|---|---|---|
| <img src="images/outdoor_scene/base_bdda9dd073dd7966.jpg" width="200"> | <img src="images/outdoor_scene/base_15f1e83bf933af9a.jpg" width="200"> | <img src="images/outdoor_scene/base_4da8c0506ec257c3.jpg" width="200"> | <img src="images/outdoor_scene/base_cfd2ddb7d56f5f43.jpg" width="200"> |
| A vendor arranges heirloom tomatoes at a stall with a city bus visible in the background.<br>`people` `setting:urban` `time:dawn_dusk` `weather:clear` | Bicycle rack and lamp post in front of orange safety netting with puddles on the ground.<br>`people` `setting:urban` `time:day` `weather:rain` | A wedding cake table stands under a white arch on the lawn at dusk.<br>`no_people` `setting:rural` `time:dawn_dusk` `weather:clear` | Distant view of kayakers in a technical rapid section.<br>`people` `setting:nature` `time:day` `weather:clear` |

An edit chain — one base image and three edits, each applied to the base or to
an earlier turn:

| | | | |
|---|---|---|---|
| <img src="images/outdoor_scene/chain_042a7a069b66c5b3_t0.jpg" width="200"> | <img src="images/outdoor_scene/chain_042a7a069b66c5b3_t1.jpg" width="200"> | <img src="images/outdoor_scene/chain_042a7a069b66c5b3_t2.jpg" width="200"> | <img src="images/outdoor_scene/chain_042a7a069b66c5b3_t3.jpg" width="200"> |
| **base image** | *edit 1 of turn 0*<br>Change the time of day to high noon with the sun directly overhead casting minimal shadows. | *edit 2 of turn 1*<br>Add a white 'Dune Protection' stenciled message painted on the boardwalk surface in the foreground. | *edit 3 of turn 2*<br>Increase the number of people to a moderate flow of walkers replacing the joggers. |

## Satellite & Aerial Views

`liquidrandom.aerial_view()` — 20,340 images.
Tags: `view:satellite`, `view:drone`, `view:aircraft`, `terrain:urban`, `terrain:agricultural`, `terrain:forest`, `terrain:water`, `terrain:desert`, `terrain:mountain`, `time:day`, `time:night`

| | | | |
|---|---|---|---|
| <img src="images/aerial_view/base_81485d346dae2a3e.jpg" width="200"> | <img src="images/aerial_view/base_83bbfe8e45778b1c.jpg" width="200"> | <img src="images/aerial_view/base_ac878bc19676c365.jpg" width="200"> | <img src="images/aerial_view/base_798ad5d26ddde5d8.jpg" width="200"> |
| A top-down satellite view of an artificial island airport with green perimeter landscaping.<br>`view:satellite` `terrain:water` `time:day` | A satellite image of an arid sandy barrier chain with stark sediment patterns.<br>`view:satellite` `terrain:desert` `time:day` | Drone view of greenhouse maintenance with rolled plastic covers.<br>`view:drone` `terrain:agricultural` `time:day` | Satellite imagery displays a multi-layered highway cutting through a dense grid of high-rises under midday sun.<br>`view:satellite` `terrain:urban` `time:day` |

An edit chain — one base image and three edits, each applied to the base or to
an earlier turn:

| | | | |
|---|---|---|---|
| <img src="images/aerial_view/chain_543647922f273956_t0.jpg" width="200"> | <img src="images/aerial_view/chain_543647922f273956_t1.jpg" width="200"> | <img src="images/aerial_view/chain_543647922f273956_t2.jpg" width="200"> | <img src="images/aerial_view/chain_543647922f273956_t3.jpg" width="200"> |
| **base image** | *edit 1 of turn 0*<br>Change the lighting to night time, darkening the ocean and highlighting reef textures with moonlight. | *edit 2 of turn 0*<br>Add a layer of thin cirrus haze over the entire scene. | *edit 3 of turn 0*<br>Add a new concrete pier extending from the western sandbank into the channel. |

## Agricultural Images

`liquidrandom.agriculture()` — 20,357 images.
Tags: `people`, `no_people`, `subject:crops`, `subject:livestock`, `subject:machinery`, `subject:pests`, `subject:infrastructure`, `season:spring`, `season:summer`, `season:autumn`, `season:winter`

| | | | |
|---|---|---|---|
| <img src="images/agriculture/base_26ae0a70e096862c.jpg" width="200"> | <img src="images/agriculture/base_34cc34c217238204.jpg" width="200"> | <img src="images/agriculture/base_d6d9eccaf8853f19.jpg" width="200"> | <img src="images/agriculture/base_7fb7b51eb9da4a6b.jpg" width="200"> |
| A blue drone is observed from inside a tractor cab while operating near mature corn at dusk.<br>`no_people` `subject:machinery` `season:autumn` | Aerial view of the harvester moving along rows with clean carrots on the conveyor.<br>`no_people` `subject:crops` `season:autumn` | A side view from the boat gunwale shows the seed rope entering the water near an orange buoy.<br>`people` `subject:infrastructure` `season:summer` | Profile view of a sow nursing piglets during twilight with blue ambient light.<br>`no_people` `subject:livestock` `season:autumn` |

An edit chain — one base image and three edits, each applied to the base or to
an earlier turn:

| | | | |
|---|---|---|---|
| <img src="images/agriculture/chain_06d8319db4d0aea5_t0.jpg" width="200"> | <img src="images/agriculture/chain_06d8319db4d0aea5_t1.jpg" width="200"> | <img src="images/agriculture/chain_06d8319db4d0aea5_t2.jpg" width="200"> | <img src="images/agriculture/chain_06d8319db4d0aea5_t3.jpg" width="200"> |
| **base image** | *edit 1 of turn 0*<br>Add heavy accumulations of dried slime trails connecting multiple leaves across the row. | *edit 2 of turn 1*<br>Change the lighting from natural dawn light to artificial purple and blue LED grow lights suspended above the plants. | *edit 3 of turn 2*<br>Change the camera angle from an eye-level row view to an extreme macro close-up of a single leaf margin. |

## Factory & Industrial Images

`liquidrandom.industrial()` — 20,060 images.
Tags: `people`, `no_people`, `subject:machinery`, `subject:robots`, `subject:assembly_line`, `subject:warehouse`, `subject:exterior`, `lighting:natural`, `lighting:artificial`, `lighting:dim`

| | | | |
|---|---|---|---|
| <img src="images/industrial/base_a0250d3b8673adbd.jpg" width="200"> | <img src="images/industrial/base_7cc1df090f937f15.jpg" width="200"> | <img src="images/industrial/base_743b816d04cbd1bf.jpg" width="200"> | <img src="images/industrial/base_6733c4cd5b2f9353.jpg" width="200"> |
| A low-angle view captures the gantry beam of an SMT machine under blue industrial lighting.<br>`no_people` `subject:machinery` `lighting:artificial` | Dark charcoal turbine enclosure at a coastal utility station during dawn.<br>`no_people` `subject:exterior` `lighting:natural` | A wide view of a bottling line capping assembly with a stainless steel filler bowl and safety fencing.<br>`people` `subject:assembly_line` `lighting:artificial` | Angled close-up of a pressure gauge interface with serial number engraving under sterile white light.<br>`people` `subject:machinery` `lighting:artificial` |

An edit chain — one base image and three edits, each applied to the base or to
an earlier turn:

| | | | |
|---|---|---|---|
| <img src="images/industrial/chain_5ea0fe48fad16dcf_t0.jpg" width="200"> | <img src="images/industrial/chain_5ea0fe48fad16dcf_t1.jpg" width="200"> | <img src="images/industrial/chain_5ea0fe48fad16dcf_t2.jpg" width="200"> | <img src="images/industrial/chain_5ea0fe48fad16dcf_t3.jpg" width="200"> |
| **base image** | *edit 1 of turn 0*<br>Change the time of day to dawn, with cool blue natural light mixing with the warm artificial interior lights through high windows. | *edit 2 of turn 1*<br>Change the machine state to under maintenance with the side access panels removed and internal wiring exposed. | *edit 3 of turn 0*<br>Change the dominant color of the crane body from industrial grey to bright safety yellow. |

## Automotive Images

`liquidrandom.automotive()` — 20,258 images.
Tags: `people`, `no_people`, `view:exterior`, `view:in_cabin`, `view:road`, `view:traffic`, `time:day`, `time:night`, `time:dawn_dusk`

| | | | |
|---|---|---|---|
| <img src="images/automotive/base_16f3d5ad1282e9c7.jpg" width="200"> | <img src="images/automotive/base_18d01831069470b2.jpg" width="200"> | <img src="images/automotive/base_f27d936e2fe01b25.jpg" width="200"> | <img src="images/automotive/base_fcf9d72f3c354a4b.jpg" width="200"> |
| A wide shot from the dashboard shows the passenger pointing at the route during twilight.<br>`people` `view:in_cabin` `time:dawn_dusk` | Daylight interior view of a driver and passengers approaching a tunnel entrance.<br>`people` `view:in_cabin` `time:day` | A close-up highlights the charging station touchscreen interface with a blurred hand holding the plug in the foreground.<br>`people` `view:exterior` `time:dawn_dusk` | Driver adjusting mirror under moonlight and street lamps with dark dashboard.<br>`people` `view:in_cabin` `time:night` |

An edit chain — one base image and three edits, each applied to the base or to
an earlier turn:

| | | | |
|---|---|---|---|
| <img src="images/automotive/chain_e80ee44ec9cd87b3_t0.jpg" width="200"> | <img src="images/automotive/chain_e80ee44ec9cd87b3_t1.jpg" width="200"> | <img src="images/automotive/chain_e80ee44ec9cd87b3_t2.jpg" width="200"> | <img src="images/automotive/chain_e80ee44ec9cd87b3_t3.jpg" width="200"> |
| **base image** | *edit 1 of turn 0*<br>Change the visible traffic outside to congested heavy traffic with brake lights glowing. | *edit 2 of turn 0*<br>Change the weather to a clear sunny day with bright natural lighting. | *edit 3 of turn 2*<br>Apply realistic capture artifacts including motion blur and high ISO noise. |

## UI & UX Screenshots

`liquidrandom.ui_screenshot()` — 20,367 images.
Tags: `platform:mobile`, `platform:desktop`, `platform:tablet`, `theme:light`, `theme:dark`, `ui_state:normal`, `ui_state:error`, `ui_state:empty`, `ui_state:loading`

| | | | |
|---|---|---|---|
| <img src="images/ui_screenshot/base_513f775effad336d.jpg" width="200"> | <img src="images/ui_screenshot/base_50e92f9dbb24eeb0.jpg" width="200"> | <img src="images/ui_screenshot/base_1ec96fbf40d5404b.jpg" width="200"> | <img src="images/ui_screenshot/base_0de06ba6cf325233.jpg" width="200"> |
| A mobile-view kiosk screen listing family bundle packages vertically.<br>`platform:mobile` `theme:light` `ui_state:normal` | A white background dashboard with purple accents and a confidence interval band on the scatter plot.<br>`platform:desktop` `theme:light` `ui_state:normal` | A dark theme mobile screen illustrating USB-C key insertion with a timeout counter.<br>`platform:mobile` `theme:dark` `ui_state:normal` | Dark mode tablet library showing recently added and favorite books.<br>`platform:tablet` `theme:dark` `ui_state:normal` |

An edit chain — one base image and three edits, each applied to the base or to
an earlier turn:

| | | | |
|---|---|---|---|
| <img src="images/ui_screenshot/chain_c7cbad1f5702fc88_t0.jpg" width="200"> | <img src="images/ui_screenshot/chain_c7cbad1f5702fc88_t1.jpg" width="200"> | <img src="images/ui_screenshot/chain_c7cbad1f5702fc88_t2.jpg" width="200"> | <img src="images/ui_screenshot/chain_c7cbad1f5702fc88_t3.jpg" width="200"> |
| **base image** | *edit 1 of turn 0*<br>Change the high contrast red accents to a deep purple color scheme for the chart bars and alerts. | *edit 2 of turn 0*<br>Translate all visible UI text and tooltip content into Japanese characters. | *edit 3 of turn 2*<br>Switch the view to an empty state showing a 'No Data Available' illustration in the center. |

## Documents & OCR

`liquidrandom.document()` — 20,222 images.
Tags: `medium:printed`, `medium:handwritten`, `medium:mixed`, `capture:scan`, `capture:photo`, `quality:clean`, `quality:degraded`

| | | | |
|---|---|---|---|
| <img src="images/document/base_a3192b0e38229c7f.jpg" width="200"> | <img src="images/document/base_dbc37cdc52a2987e.jpg" width="200"> | <img src="images/document/base_367a7a6aeb65518b.jpg" width="200"> | <img src="images/document/base_c5b042fa72fad78e.jpg" width="200"> |
| A light gray shipping label on beige cardboard with a torn 2D barcode and grease residue.<br>`medium:printed` `capture:photo` `quality:degraded` | A scanned corporate invoice featuring bilingual English and Arabic text with double-line borders.<br>`medium:printed` `capture:scan` `quality:clean` | A close-up photo of an amended tax return calculation section.<br>`medium:mixed` `capture:photo` `quality:clean` | A crumpled diary note on blue-lined paper under fluorescent lighting.<br>`medium:handwritten` `capture:photo` `quality:degraded` |

An edit chain — one base image and three edits, each applied to the base or to
an earlier turn:

| | | | |
|---|---|---|---|
| <img src="images/document/chain_0c26a586d457833a_t0.jpg" width="200"> | <img src="images/document/chain_0c26a586d457833a_t1.jpg" width="200"> | <img src="images/document/chain_0c26a586d457833a_t2.jpg" width="200"> | <img src="images/document/chain_0c26a586d457833a_t3.jpg" width="200"> |
| **base image** | *edit 1 of turn 0*<br>Shift the camera perspective to a low angle looking up at the signage, emphasizing the height of the wall and the sky background. | *edit 2 of turn 0*<br>Translate all menu text to French and convert currency values from USD to Euro, adapting item names to local cuisine terms. | *edit 3 of turn 2*<br>Apply digital noise and fading effects to simulate a low-resolution scan, reducing overall sharpness and contrast. |

## Charts & Diagrams

`liquidrandom.chart()` — 20,295 images.
Tags: `form:bar`, `form:line`, `form:pie`, `form:scatter`, `form:flowchart`, `form:table`, `form:infographic`, `form:diagram`, `style:clean`, `style:hand_drawn`

| | | | |
|---|---|---|---|
| <img src="images/chart/base_dbe1cd8890518861.jpg" width="200"> | <img src="images/chart/base_aea008eb78b9860b.jpg" width="200"> | <img src="images/chart/base_600dbf7ca123c167.jpg" width="200"> | <img src="images/chart/base_c218937d945e513c.jpg" width="200"> |
| A scientific scatter plot comparing drug clearance against weight separated by patient gender.<br>`form:scatter` `style:clean` | A network graph diagram illustrating transit times from multiple farms to a central quality hub.<br>`form:diagram` `style:clean` | A clean vertical grouped bar chart showing quarterly sales across five regions with growth annotations.<br>`form:bar` `style:clean` | A donut-style pie chart showing wearable unit share with percentages inside slices.<br>`form:pie` `style:clean` |

An edit chain — one base image and three edits, each applied to the base or to
an earlier turn:

| | | | |
|---|---|---|---|
| <img src="images/chart/chain_177edbcae2fb652a_t0.jpg" width="200"> | <img src="images/chart/chain_177edbcae2fb652a_t1.jpg" width="200"> | <img src="images/chart/chain_177edbcae2fb652a_t2.jpg" width="200"> | <img src="images/chart/chain_177edbcae2fb652a_t3.jpg" width="200"> |
| **base image** | *edit 1 of turn 0*<br>Render the chart in a hand-drawn whiteboard style with sketchy lines and marker-like textures. | *edit 2 of turn 1*<br>Adjust the data values to reflect Q1 2024 capacity figures and update the central total to 475 GWh. | *edit 3 of turn 0*<br>Add a new data series segment for 'Northvolt' with 5% share and update the legend accordingly. |

## Retail & Products

`liquidrandom.retail_product()` — 20,499 images.
Tags: `people`, `no_people`, `presentation:studio`, `presentation:shelf`, `presentation:packaging`, `presentation:lifestyle`

| | | | |
|---|---|---|---|
| <img src="images/retail_product/base_441a61b26971fbf5.jpg" width="200"> | <img src="images/retail_product/base_3d039d5215fcb437.jpg" width="200"> | <img src="images/retail_product/base_bfcdeb8781773035.jpg" width="200"> | <img src="images/retail_product/base_cffb15165fe21ac4.jpg" width="200"> |
| High-angle view of luxury satin gowns on a boutique rack under ambient lighting.<br>`no_people` `presentation:shelf` | Brightly lit retail end-cap displaying smart home hubs and sorted LED bulb boxes.<br>`people` `presentation:shelf` | A reclaimed wood veneer convertible bed and desk unit is shown under overcast daylight with linoleum flooring.<br>`no_people` `presentation:studio` | A reflection shot of silver cylindrical boxes on a copper counter with a digital display.<br>`people` `presentation:packaging` |

An edit chain — one base image and three edits, each applied to the base or to
an earlier turn:

| | | | |
|---|---|---|---|
| <img src="images/retail_product/chain_4960d1f0a6d945c2_t0.jpg" width="200"> | <img src="images/retail_product/chain_4960d1f0a6d945c2_t1.jpg" width="200"> | <img src="images/retail_product/chain_4960d1f0a6d945c2_t2.jpg" width="200"> | <img src="images/retail_product/chain_4960d1f0a6d945c2_t3.jpg" width="200"> |
| **base image** | *edit 1 of turn 0*<br>Add a closed silver laptop to the center of the glass desk surface. | *edit 2 of turn 0*<br>Change the stock level on the bookshelf to appear messy and disorganized. | *edit 3 of turn 0*<br>Add a person standing behind the desk looking at the bookshelf. |

## Food & Cooking

`liquidrandom.food()` — 20,004 images.
Tags: `people`, `no_people`, `stage:raw`, `stage:cooking`, `stage:plated`, `setting:home`, `setting:restaurant`, `setting:studio`

| | | | |
|---|---|---|---|
| <img src="images/food/base_23c5d7af1870e226.jpg" width="200"> | <img src="images/food/base_02f6a98adc175993.jpg" width="200"> | <img src="images/food/base_104a6fff6eb57642.jpg" width="200"> | <img src="images/food/base_698326b860993e2d.jpg" width="200"> |
| Ossobuco served in a cast iron pot with a spoon scooping marrow in a home kitchen.<br>`no_people` `stage:plated` `setting:home` | A night market display of tropical fruits under fluorescent lights with visible steam.<br>`people` `stage:raw` `setting:restaurant` | A chef stretches pizza dough outdoors near a brick oven during golden hour.<br>`people` `stage:cooking` `setting:restaurant` | An eye-level view of a chef garnishing a scallop in a warm restaurant kitchen.<br>`people` `stage:plated` `setting:restaurant` |

An edit chain — one base image and three edits, each applied to the base or to
an earlier turn:

| | | | |
|---|---|---|---|
| <img src="images/food/chain_c0fc98c9842337d6_t0.jpg" width="200"> | <img src="images/food/chain_c0fc98c9842337d6_t1.jpg" width="200"> | <img src="images/food/chain_c0fc98c9842337d6_t2.jpg" width="200"> | <img src="images/food/chain_c0fc98c9842337d6_t3.jpg" width="200"> |
| **base image** | *edit 1 of turn 0*<br>Change the whole wheat dough to white sourdough dough with a lighter color. | *edit 2 of turn 0*<br>Add a dusting of rice flour on top of the dough instead of plain wheat flour. | *edit 3 of turn 2*<br>Change the twilight blue hour lighting to bright midday sun. |

