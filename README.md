# Final Project: Stress Testing a Paper From NeurIPS Titled "Parts of Speech–Grounded Subspaces in Vision-Language Models".

## 1. Introduction

### 1.1. Original Paper Brief Description
Oldfield et al., (2023) introduced an innovative approach to enhancing vision-language models by structuring latent image representations around the linguistic concept of parts of speech (PoS). Specifically, it addresses the challenge of entanglement in visual attributes within CLIP image representations, which often leads to biases toward certain visual properties like objects or actions. The authors propose a method to disentangle these representations by associating different visual modalities with specific parts of speech (PoS), such as nouns with objects and adjectives with appearances. The effectiveness of this approach is demonstrated in the paper through both qualitative and quantitative evaluations. 

### 1.2. Problem Domain
This project focuses on stress testing the novel methodologies proposed by Oldfield et al., (2023) in the NeurIPS paper. The primary goal of this project is to evaluate the robustness and effectiveness of these methodologies under different conditions. Moreover, this stress testing specifically targets the qualitative evaluation of the model, assessing the visual quality and relevance of images generated in response to complex textual prompts to determine the model's interpretative capabilities. Additionally, developers and researchers in the field of AI, particularly those involved in enhancing multimodal interaction systems, might find this stress testing particularly usefull for validating and refining vision-language models.

## 2. Method
### 2.1. High-Level Approach
The methodology employed in this project is based on a thorough replication of the experiments detailed by Oldfield et al., (2023). Initially, the codebase from the authors' GitHub repository was leveraged to run the experiment notebook, specifically **pos.ipynb**, using Google Colab's A100 GPU to manage the substantial computational demands—approximately 32 GB of GPU RAM. To establish a baseline, images were first generated using the exact prompts provided in the original paper to confirm the operational integrity and expected performance of the model. Subsequently, the scope of testing was expanded by introducing a variety of prompts designed to probe and challenge the model's capabilities further. In the context of this project, no external datasets were utilized since the primary focus was on evaluating the model’s response to a diverse set of prompts. 

### 2.2. Complexity of the Approach
The execution of the experiments was straightforward in terms of technical complexity; however, the lack of comprehensive instructions and sparse commenting within the code posed challenges in replicating the original experiments. Additionally, the computational resources required for this project were substantial. Utilizing Google Colab's A100 GPU, which is necessary to handle the almost 32 of GPU RAM required by the model, entails significant computational costs and time.

<img width="333" alt="Screenshot 2024-04-25 at 13 32 31" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/7d78b91e-1ebc-419d-a46b-3388384b821c">


### 2.3. Categories of Stress Testing
The stress testing encompassed evaluations across all three original categories:
1. **Adj/Noun Subspace Projection**: This test examined how well the model could handle and differentiate between adjectives and nouns in generating images that accurately reflect the described attributes and objects. In addition, how the model handels a prompt with multiple adjuctives (coordinate adjectives).
2. **Style-Blocking Adjective Projection**: In this category, the focus was on the model’s ability to block out specific stylistic adjectives from influencing the visual output.
3. **Custom Subspace Projection**: Two custom subspaces—'fire' and 'interstellar'—were added to the dectionary.

### 2.4. Custom Dictionaries Added For Test

    'fire': {
        'custom_dict': list(set([
            "Wildfire", "House fire", "Bonfire", "Campfire", "Backdraft", "Firestorm", "Pyrotechnics", "Fireworks",
            "Burning man", "Fire breathing", "Flame thrower", "Molotov cocktail", "Forest fire", "Bushfire",
            "Firefighter", "Fire engine", "Hose stream", "Fire hydrant", "Fire alarm", "Smoke detector",
            "Evacuation", "Escape route", "Fire escape", "Burn", "Char", "Scorch", "Singe", "Blaze",
            "Inferno", "Conflagration", "Arson", "Flammable", "Combustible", "Ignite", "Kindle", "Smolder",
            "Ash", "Ember", "Spark", "Flint", "Match", "Lighter", "Candle", "Lantern", "Torch", "Oil lamp",
            "Heatwave", "Hotspot", "Flashover", "Meltdown", "Incendiary", "Explosion", "Fireball", "Burnout",
            "Fire ring", "Fire dance", "Fire show", "Fire art", "Fire performance", "Fire festival", "Fire ceremony",
            "Cremation", "Pyre", "Sacrificial fire", "Eternal flame", "Phoenix rising", "Fiery sunset", "Lava flow",
            "Volcano", "Dragon breath", "Hellfire", "Fiery furnace", "Burning bridge", "Fire line", "Fire barrier",
            "Firebreak", "Fireproof", "Fire resistance", "Heat resistant", "Thermal barrier", "Fire control",
            "Fire prevention", "Fire safety", "Fire drill", "Fire hazard", "Fire risk", "Fire watch", "Fire guard",
            "Fire blanket", "Fire extinguisher", "Fire retardant", "Fire suppression", "Firefighting foam",
            "Fire insurance", "Fire damage", "Fire claim", "Fire loss", "Fire recovery", "Fire investigation",
            "Fire origin", "Fire cause", "Fire effect", "Fire aftermath", "Fire debris", "Fire cleanup", "Fire rebuild"
        ]))
    },
    'interstellar': {
        'custom_dict': list(set([
            "Black hole", "Neutron star", "Pulsar", "Quasar", "Supernova", "Nebula", "Galaxy", "Exoplanet",
            "Space station", "Spaceship", "Rocket launch", "Asteroid belt", "Meteor shower", "Comet tail",
            "Alien civilization", "UFO sighting", "Zero gravity", "Spacewalk", "Moon landing", "Martian landscape",
            "Interstellar travel", "Wormhole traversal", "Time dilation", "Light year", "Dark matter",
            "Dark energy", "Space-time continuum", "Parallel universe", "Multiverse", "Cosmic microwave background",
            "Big Bang", "Cosmic rays", "Solar flare", "Aurora", "Ionosphere", "Orbit", "Escape velocity",
            "Satellite", "Telescope", "Observatory", "Astrophysics", "Astronomy", "Cosmology", "Space exploration",
            "Space colonization", "Habitable zone", "Goldilocks planet", "Terraforming", "Astrobiology",
            "Space habitat", "Biosphere", "Extraterrestrial life", "First contact", "Alien abduction",
            "Space opera", "Sci-fi epic", "Interstellar war", "Space battle", "Starfleet", "Guardians of the galaxy",
            "Star Wars", "Star Trek", "The Expanse", "Foundation", "Dune", "Gravity", "Interstellar",
            "Cosmic voyage", "Space odyssey", "Galactic empire", "Rebel alliance", "Space pirate",
            "Alien queen", "Space diplomacy", "Galactic council", "Space mission", "Launch pad", "Mission control",
            "Astronaut training", "Space suit", "Life support system", "Oxygen generator", "Hydroponics",
            "Astro-navigation", "Stellar cartography", "Quantum entanglement", "Warp drive", "Hyperspace",
            "Subspace communication", "Photon torpedo", "Plasma cannon", "Shield generator", "Cloaking device",
            "Alien artifact", "Space relic", "Mysterious monolith", "Cosmic anomaly", "Gravitational lensing"
        ]))
    }

### 2.5. Adj-Noun Overlap Assessment
To investigate the overlap between nouns and adjectives within subspace, I employed a method involving the projection of words embeddings (for both adjs and nouns) onto a noun subspace followed by the evaluation of cosine similarities. This approach aims to identify pairs of adjectives and nouns that, despite their distinct categorical attributes, share a high degree of similarity in their vector representations when viewed within the confines of the noun subspace. Moreover, each embedding vector, representing nouns **(N)** and adjectives **(A)**, was first normalized to unit length to standardize their magnitudes, facilitating a fair comparison. These normalized vectors were then projected onto the noun subspace **(WN)**, calculated as the matrix product of the noun embeddings and their transpositions, resulting with vectors **N_in_N** for nouns and **A_in_N** for adjectives.

#### 2.5.1. Cosine Similarity Calculation
```
def find_high_similarity_pairs(N, A, WN, threshold=0.90):
    # Normalize the embeddings
    N_norm = N / N.norm(dim=1, keepdim=True)
    A_norm = A / A.norm(dim=1, keepdim=True)

    # Project both nouns and adjectives onto the noun subspace
    N_in_N = torch.mm(N_norm, torch.mm(WN, WN.transpose(0, 1)))
    A_in_N = torch.mm(A_norm, torch.mm(WN, WN.transpose(0, 1)))

    # Renormalize projections
    N_in_N = N_in_N / N_in_N.norm(dim=1, keepdim=True)
    A_in_N = A_in_N / A_in_N.norm(dim=1, keepdim=True)

    # Calculate cosine similarities
    cosine_similarities = torch.mm(N_in_N, A_in_N.t())

    # Filter pairs with cosine similarity over the threshold
    high_sim_indices = torch.where(cosine_similarities > threshold)
    return high_sim_indices, cosine_similarities
```
Post-projection, cosine similarities were computed between all noun and adjective vectors now represented in the noun subspace. Pairs exhibiting similarities above a defined threshold (e.g., 0.90) were saved for further analysis. This high threshold was set to ensure only the most similar pairs were considered, aiming to capture those adjective-noun combinations that, unexpectedly, align closely within a noun-centric semantic field.

#### 2.5.2. Sentiment Analysis

```
from transformers import pipeline
sentiment_classifier = pipeline('sentiment-analysis')
def get_sentiment(word):
    result = sentiment_classifier(word)
    sentiment = result[0]['label'].lower()
    return sentiment
```

To add a layer of interpretative depth to the findings, I integrated sentiment analysis into the examination of adjective-noun pairs. Utilizing pre-defined sentiment analysis tool (as shown above), each word in the identified pairs was classified as positive, negative, or neutral. This classification is used to understand whether the semantic similarity also extended to some connotations, which could be pivotal in nuanced linguistic or contextual applications.


## 3. Results
### 3.1. Baseline Results
To ensure the integrity and functionality of the model, initial tests replicated the experiments described in Oldfield et al., (2023). These baseline tests used the same prompts from the paper to generate images, allowing for a direct comparison with the published results. The outcomes from this phase were consistent with those documented in the NeurIPS paper, confirming that the model was operational as expected. This validation step was crucial for establishing a reliable foundation for the subsequent stress testing.

**Prompt 1**: "A photo of a pink croissant"

**Result from the paper:**

<img width="987" alt="Screenshot 2024-04-25 at 16 37 47" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/bfa3eb39-7799-4d40-bd83-6f5efcd8a093">

**Result from the test:**

<img width="815" alt="Screenshot 2024-04-25 at 16 38 17" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/4b1ba195-4560-462d-8e20-8e0c503e2548">


**Prompt 2**: "A photo of Karl Marx in a disney movie"

**Result from the paper:**
<img width="1199" alt="Screenshot 2024-04-25 at 16 39 35" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/720466a9-3a9c-4218-b1ad-1cd46a40ee31">


**Result from the test:**
<img width="1269" alt="Screenshot 2024-04-25 at 13 18 08" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/a288c918-bf19-4eb0-b52f-e9c3fdbf72e6">

### 3.2. Adj/Noun Subspace Projection Results

The Adj/Noun Subspace Projection was designed to evaluate the model's ability to segregate and selectively remove noun and adjective descriptors from the prompts provided. However, the results revealed some inconsistencies:

<img width="825" alt="Screenshot 2024-04-25 at 16 40 22" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/ab394064-a8cf-4a49-9493-b7db18999abb">

"A photo of a green dying tree": The model still displayed images of trees under Noun subspace orthogonal projection, contrary to the claim that it successfully removes object content from embeddings. Moreover, the presence of green trees in the Adjective subspace orthogonal projection indicates a failure to disregard visual appearances, which should have been removed according to the model's specifications.

<img width="810" alt="Screenshot 2024-04-25 at 16 40 42" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/0df4d50a-aff2-40ad-9705-fda90d5c3d0c">

"A photo of a blue rusty car": This test showed a partial success with the adjective "rusty" being depicted in both projections but failed to eliminate the color "blue" in the Adj subspace orthogonal projection and to eliminate the noun "car" in the noun subspace orthogonal projection.

<img width="813" alt="Screenshot 2024-04-25 at 16 40 57" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/3548b7c6-aa93-4b28-9d39-9858d544ca69">

"A photo of an old rusty car": Similar to the previous example, the model did not effectively differentiate between the adjectives in either subspace projection, with "rusty" and "old" features still prominent. In addition, the "car" object shouldn't exist in the noun subspace orthogonal projection.

<img width="815" alt="Screenshot 2024-04-25 at 16 41 18" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/ef7b3548-8fa5-4b56-8a0d-0267881c46fb">

"A photo of an old car": The failure to remove the noun "car" in the Noun subspace and the adjective "old" in the Adj. subspace orthogonal projection highlights a significant challenge in the model's processing capabilities.

<img width="811" alt="Screenshot 2024-04-25 at 16 41 35" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/4da25fe0-7449-4298-8c4e-7dbda5df2805">

"A photo of a multicoloured car": (Just like the example from the paper but instead of "penguin" I used "car") The model inconsistently handled the adjective "multicoloured" in both projections, sometimes ignoring it but often retaining colorful elements. The same goes to the noun subspace orthogonal projection, the car (noun) still visible in some images.

<img width="813" alt="Screenshot 2024-04-25 at 16 41 53" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/68a2c991-7dc1-444f-a89d-315a4d9b77c7">

"A photo of snowy Big Apple": This test raised questions about the model's handling of compound nouns and adjectives, with neither "snowy" nor "Big Apple" being effectively ignored in their respective projections.

<img width="809" alt="Screenshot 2024-04-25 at 16 42 31" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/772cefa5-b866-41ee-bfb8-d3700f3b4c00">
<img width="809" alt="Screenshot 2024-04-25 at 16 42 48" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/27abdcd0-8148-439c-b3ef-d20d5d0db23b">

"A photo of green grass" and "A photo of a snowy mountain": Both showed a failure to ignore the nouns and adjectives respectively.

More results below:
<img width="1179" alt="Screenshot 2024-04-25 at 16 44 54" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/47949e76-faec-4671-b5a8-ac0881999532">

<img width="1221" alt="Screenshot 2024-04-25 at 16 44 24" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/ab3baa22-77be-426c-ad17-070f842ed7fd">

<img width="811" alt="Screenshot 2024-04-25 at 16 43 12" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/5a3f6a5c-7c1d-4319-af8c-108d81984938">

<img width="816" alt="Screenshot 2024-04-25 at 16 43 23" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/35b3ea85-ac3d-4ffc-9ceb-1a4a0ef29c54">

<img width="820" alt="Screenshot 2024-04-25 at 16 43 32" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/6aad2372-0eea-4d1e-ad9e-d0d3525d3ad8">

<img width="815" alt="Screenshot 2024-04-25 at 16 43 51" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/31a76a60-f127-4884-b8b9-00bc5345fd8c">

<img width="815" alt="Screenshot 2024-04-25 at 16 44 03" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/4c0a47af-9093-455c-9212-a04f8ebadbf5">


### 3.3. Style-Blocking Adjective Projection Results

This section focused on the model's capability to block specific styles associated with adjectives in prompts:

<img width="1052" alt="Screenshot 2024-04-25 at 16 45 37" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/08bf1cf0-ef85-4793-8342-94807d048065">

"A red Gauguin painting of Einstein": The model successfully blocked the artistic style.

<img width="852" alt="Screenshot 2024-04-25 at 16 45 51" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/0fb4194d-3054-4cfe-bb0e-8b27aa4655cb">

"A circular painting of the Eiffel Tower in the style of Rothko": (The word 'circular' was added to the original prompt in the paper) While attempting to block the style, the model also  blocked the depiction of the Eiffel Tower, indicating an overextension of the style-blocking mechanism.

### 3.4. Custom Subspace Projection Results

The introduction of custom themes like "Fire" and "Interstellar" was intended to assess the model's flexibility and adaptability:


#### Theme: Fire
<img width="1084" alt="Screenshot 2024-04-25 at 16 46 24" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/4438d231-6784-433c-832d-a16a45419ee2">

1. "Ancient forest caught in a devastating wildfire, with flames reaching sky-high, illuminating the night."

<img width="1070" alt="Screenshot 2024-04-25 at 16 46 54" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/60a30870-ffbe-40dd-aaa0-98cf38b03543">

2. "A medieval castle engulfed in fierce flames during a siege, with archers still firing arrows through the smoke and fire".

The model effectively blocked the fire theme from all images in the prompts above, demonstrating a successful application of the custom subspace projection.

#### Theme: Interstellar

<img width="1044" alt="Screenshot 2024-04-25 at 16 47 34" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/1a683f87-788f-48f2-becc-16ff9b77f28f">

"A plane going into a Black hole." the model showed images of bugs. Even though the word 'plane' was not in the interstellar part of the Custom Subspace dictionary, yet, it wasn't in the generated images.

### 3.5. Results Summary
The test results reveal that while the model shows some capability to adhere to the framework of subspace projections, there are notable inconsistencies and failures, particularly in handling complex, multi-component, coordinate adjectives prompts. These issues highlight the need for further refinement and possibly more sophisticated training or projection mechanisms to achieve reliable disentanglement of visual attributes as claimed.

## 4. Results Analysis


![sssss](https://github.com/AzizGW/CvFinalProject/assets/119353586/f1dd16a9-0c77-4c9f-b729-1e62de5bffdb)

### 4.1. Adj-Noun Overlap Sample output
```
unaffixed (negative) + tuxedo (positive) - Cosine Score: 0.95
verrucose (negative) + pyroelectricity (positive) - Cosine Score: 0.96
glossopharyngeal (negative) + direct examination (positive) - Cosine Score: 1.00
ungarbed (negative) + Otto von Bismarck (positive) - Cosine Score: 1.00
lxii (positive) + barium (negative) - Cosine Score: 1.00
gregarious (positive) + St. Gregory of Nazianzen (negative) - Cosine Score: 1.00
geosynchronous (positive) + systematist (positive) - Cosine Score: 1.00
irritated (negative) + attendance check (positive) - Cosine Score: 0.98
sightly (positive) + cerebellar hemisphere (negative) - Cosine Score: 1.00
unpracticed (negative) + gasoline bomb (negative) - Cosine Score: 0.96
gangling (negative) + great auk (positive) - Cosine Score: 1.00
square-shouldered (positive) + razorback hog (negative) - Cosine Score: 1.00
viscoelastic (positive) + relaxation (positive) - Cosine Score: 1.00
aimless (negative) + June 23 (positive) - Cosine Score: 1.00
out of true (positive) + Anigozanthus (positive) - Cosine Score: 1.00
tripinnate (negative) + diaper rash (negative) - Cosine Score: 1.00
sere (negative) + Richard II (positive) - Cosine Score: 0.96
inscrutable (negative) + Myrtillocactus geometrizans (positive) - Cosine Score: 1.00
stylish (positive) + Atriplex lentiformis (negative) - Cosine Score: 0.96
barometrical (positive) + Trojan horse (negative) - Cosine Score: 0.95
city-bred (positive) + Lepas (positive) - Cosine Score: 1.00
on the fence (positive) + synergism (positive) - Cosine Score: 1.00
satiable (positive) + Gypsy Rose Lee (positive) - Cosine Score: 0.96
nursed (negative) + angiologist (negative) - Cosine Score: 0.96
apopemptic (negative) + youth movement (positive) - Cosine Score: 0.95
protuberant (positive) + stableman (positive) - Cosine Score: 1.00
mass-spectrometric (positive) + niece (positive) - Cosine Score: 1.00
telescoped (positive) + Didrikson (positive) - Cosine Score: 1.00
in good taste (positive) + acoustic projection (positive) - Cosine Score: 0.96
catadromous (negative) + marabou (positive) - Cosine Score: 0.98
unadorned (negative) + Bruxelles (positive) - Cosine Score: 0.97
stubby (negative) + dog flea (negative) - Cosine Score: 1.00
hueless (negative) + photogelatin process (positive) - Cosine Score: 0.96
forgettable (negative) + ballpoint pen (positive) - Cosine Score: 1.00
6th (positive) + newsreader (negative) - Cosine Score: 0.95
metastatic (negative) + Mount Olympus (positive) - Cosine Score: 1.00
enceinte (positive) + trinitroglycerin (negative) - Cosine Score: 0.95
burbling (negative) + channelisation (negative) - Cosine Score: 0.95
dissonant (negative) + Calla palustris (negative) - Cosine Score: 0.95
ratty (negative) + wiggle nail (positive) - Cosine Score: 1.00
grimy (negative) + footwear (positive) - Cosine Score: 0.97
unrhymed (negative) + Dien Bien Phu (positive) - Cosine Score: 1.00
enclosed (positive) + Elli (positive) - Cosine Score: 0.97
forty-seven (positive) + drainage basin (positive) - Cosine Score: 0.96
purse-proud (positive) + Erigeron canadensis (positive) - Cosine Score: 0.97
white-shoe (negative) + snowboarding (positive) - Cosine Score: 0.97
deep-lobed (negative) + vertebrate foot (positive) - Cosine Score: 1.00
300 (positive) + prankishness (negative) - Cosine Score: 1.00
disappointed (negative) + infantile amaurotic idiocy (negative) - Cosine Score: 0.97
mop-headed (negative) + banded palm civet (positive) - Cosine Score: 1.00
in sight (positive) + Myrica pensylvanica (positive) - Cosine Score: 1.00
witty (positive) + mucous secretion (negative) - Cosine Score: 0.98
transgendered (negative) + genus Ipomoea (negative) - Cosine Score: 1.00
inconsequential (negative) + swamp horsetail (negative) - Cosine Score: 1.00
anal retentive (negative) + writing desk (negative) - Cosine Score: 0.97
pissed off (negative) + jet set (positive) - Cosine Score: 1.00
antrorse (negative) + dewberry bush (positive) - Cosine Score: 0.95
encouraging (positive) + Mendelsohn (positive) - Cosine Score: 0.95
moribund (negative) + bachelorette (positive) - Cosine Score: 0.96
incognizable (negative) + good guy (positive) - Cosine Score: 1.00
cumbersome (negative) + meadow rue (positive) - Cosine Score: 1.00
spring-flowering (positive) + John Rowlands (positive) - Cosine Score: 1.00
leptorrhine (negative) + pulassan (positive) - Cosine Score: 0.96
sonsy (positive) + President Garfield (positive) - Cosine Score: 1.00
subarctic (positive) + tumblebug (negative) - Cosine Score: 0.96
purplish-white (negative) + chicken cacciatora (negative) - Cosine Score: 1.00
wrapped (positive) + scrubbird (negative) - Cosine Score: 0.96
grimy (negative) + footwear (positive) - Cosine Score: 0.97
```

### 4.2. Testing The Model Again

<img width="971" alt="Screenshot 2024-05-05 at 17 47 16" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/53f58dc1-af54-4572-a3a6-829278687e8b">

**1. "A photo of purplish-white chicken cacciatora"**: The model successfully executed both the noun subspace orthogonal projection and the adjective subspace orthogonal projection without any issues.

<img width="971" alt="Screenshot 2024-05-05 at 17 47 35" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/acd9e4fc-9a1a-4e45-b428-86642ba4af0f">

**2. "A photo of wrapped scrubbird"**: Similarly to the first example, the model successfully handled the orthogonal projection for both the noun "scrubbird" and the adjective "wrapped".
<img width="971" alt="Screenshot 2024-05-05 at 17 48 06" src="https://github.com/AzizGW/CvFinalProject/assets/119353586/7c1aa662-204a-4903-981a-ec19f3f287c7">

**3. "A photo of grimy footwear"**: In this case, the model encountered difficulties specifically with the noun subspace orthogonal projection. It failed to effectively ignore the noun "footwear", resulting in some images where the noun was not correctly omitted. Conversely, in the adjective subspace orthogonal projection, the model generated images without the footwear.

### 4.2. Sentiment Correlation and Threshold Impact
The findings reveal a trend: as the cosine similarity threshold increases, so does the proportion of adjective-noun pairs sharing the same sentiment. This trend may suggests a potential alignment between semantic closeness and emotional congruence, indicating that overlaping pairs in a subspace are more likely to share the same sentiment (positive, neutral, or negative).To illustrate, at the highest observed threshold (cosine score = 1.0), 59.5% of the adj-noun pairs shared the same sentiment, demonstrating the strongest alignment both semantically and sentimentally. Moreover,this proportion remains relatively high with a slight decrease as the threshold lowers: 58.6% at >0.99, 57.8% at >0.98, and 56.1% at >0.97. The ratio continues to taper off as the threshold decreases, with 55.0% at >0.96 and exactly 50% at >0.95, reinforcing the observation that higher semantic alignment correlates with higher sentiment congruence. As I expand the scope to include pairs with lower semantic similarity (thresholds >0.90 and >0.85), the proportion of sentiment alignment drops further to 48.2% and 27.5%, respectively.

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <table>
        <thead>
            <tr>
                <th>Threshold</th>
                <th>Pairs with Similar Sentiments</th>
                <th>Total Pairs</th>
                <th>Ratio</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>= 1.0</td>
                <td>28</td>
                <td>47</td>
                <td>0.595</td>
            </tr>
            <tr>
                <td>&gt; 0.99</td>
                <td>54</td>
                <td>92</td>
                <td>0.586</td>
            </tr>
            <tr>
                <td>&gt; 0.98</td>
                <td>55</td>
                <td>95</td>
                <td>0.578</td>
            </tr>
            <tr>
                <td>&gt; 0.97</td>
                <td>64</td>
                <td>114</td>
                <td>0.561</td>
            </tr>
            <tr>
                <td>&gt; 0.96</td>
                <td>78</td>
                <td>142</td>
                <td>0.550</td>
            </tr>
            <tr>
                <td>&gt; 0.95</td>
                <td>98</td>
                <td>196</td>
                <td>0.500</td>
            </tr>
            <tr>
                <td>&gt; 0.90</td>
                <td>454</td>
                <td>941</td>
                <td>0.482</td>
            </tr>
            <tr>
                <td>&gt; 0.85</td>
                <td>963</td>
                <td>3504</td>
                <td>0.275</td>
            </tr>
        </tbody>
    </table>
</body>
</html>


## 5. Conclusion 

The evaluation of 'Parts of Speech–Grounded Subspaces in Vision-Language Models' reveals that while the model conceptually promises to disentangle visual attributes based on linguistic inputs, it struggles with consistent execution, particularly with complex prompts. In tests such as Adj/Noun Subspace Projection and Style-Blocking Adjective Projection, the model occasionally succeeded but often failed to accurately manage and block visual attributes as claimed. This inconsistency poses significant challenges for its application in professional environments that demand high precision.

Despite these challenges, it is important to recognize that a deeper understanding of the model is required. However, due to limited time and computational resources, extensive testing and exploration were not feasible at this stage. It remains unclear why the model fails in certain instances with specific nouns or adjectives (other than the limitation mentioned in the paper), suggesting that further research is necessary to fully uncover and address these limitations.


## References
1. Oldfield, J., Tzelepis, C., Panagakis, Y., Nicolaou, M., & Patras, I. (2023). Parts of Speech–Grounded Subspaces in Vision-Language Models. Advances in Neural Information Processing Systems, 36, 2700-2724.
2. Original paper code: https://github.com/james-oldfield/PoS-subspaces.
3. Google Colab.
