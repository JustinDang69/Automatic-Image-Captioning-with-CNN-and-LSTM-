# Updated CNN–LSTM Architecture (Tutor-approved)

This diagram documents the *current project design* and the two tutor-approved decoding innovations. It complements the existing `mermaid-diagram.png` (retained to preserve team contributions).

```mermaid
flowchart TD
    A["Input RGB image (224 × 224)"] --> B["VGG16 ImageNet · frozen"]
    B --> C["fc2 visual feature · 4,096-d"]
    C --> D["Dense 256 + ReLU"]
    E["Training caption prefix · startseq + words"] --> F["Tokenizer + left padding · max 39"]
    F --> G["Embedding 256 + Dropout"]
    G --> H["LSTM 256"]
    D --> I["Add · feature fusion (256-d)"]
    H --> I
    I --> J["Dense 256 + ReLU"]
    J --> K["Softmax · next-word probabilities"]
    K --> L["Greedy decoding · baseline"]
    K --> M["Beam search · baseline"]
    K --> N["Length-normalised beam search · innovation 1"]
    K --> O["Top-k sampling · innovation 2"]
    L --> P["Generated captions + held-out test BLEU-1…BLEU-4"]
    M --> P
    N --> P
    O --> P
    classDef encoder fill:#e8f4fa,stroke:#79b4d7,color:#193650
    classDef decoder fill:#f1eafa,stroke:#ae93d4,color:#193650
    classDef innovation fill:#fff1dd,stroke:#d49b4e,color:#193650
    classDef metrics fill:#e9f2f8,stroke:#a2b6c5,color:#193650
    class A,B,C,D encoder
    class E,F,G,H decoder
    class N,O innovation
    class P metrics
```

**Important distinctions:** VGG16 features are extracted once and reused; no VGG16 retraining is required. The decoder is trained using only the image-level training split, with validation for model selection. Greedy/beam search and the two innovations operate **after** softmax during inference, not as trainable LSTM layers. Compare BLEU scores *before vs after* enhancements on the same held-out test set. Top-k sampling is stochastic, so record the random seed; results need not improve BLEU.

The full graphical PNG/SVG version can be inserted into the group report. Do not confuse example captions/illustrations with actual experimental outputs.
