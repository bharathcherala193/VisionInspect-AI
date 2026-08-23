# VisionInspect AI

A defect detection system for manufacturing quality inspection, built on the MVTec Anomaly Detection dataset. It fine-tunes a ResNet50 to tell good products apart from defective ones, and it's been tested on three different product categories so far: bottles, cables, and carpet.

I built this as a portfolio project to actually go through the full loop of a real ML system — not just train a model and call it done, but deal with the stuff that comes up in practice: imbalanced data, unstable training, and figuring out when a result is genuinely good versus when you're just fooling yourself.

## Results

| Category | Accuracy | Defect Recall | Notes |
|---|---|---|---|
| Bottle | 0.98 | 0.91 | Structural defects (cracks, breaks) |
| Cable | 0.92 | 0.86 | Structural defects, but more defect types and less data per type |
| Carpet | 0.87 | 0.85 | Texture-based defects — harder problem, more on this below |

These aren't cherry-picked numbers. Every result here came from a held-out test set the model never saw during training, and I'll explain below where I had to fight for them and where I decided to stop pushing.

## Why these three categories

I didn't train on all 15 MVTec categories out of the gate. I started with `bottle` alone to get the pipeline working end to end — data splitting, training, evaluation, inference — without the complexity of switching categories getting in the way of finding real bugs. Once that worked, I picked two more categories on purpose: `cable`, because it has way more defect types (8, versus bottle's 3) and less data per type, and `carpet`, because its defects are texture-based rather than structural — a genuinely different visual problem. The idea was to see if the same pipeline held up across meaningfully different kinds of defects, not just three easy variations of the same thing.

It turned out to be the right call. The results actually tell a story: structural defects (bottle, cable) are easier for this approach, and texture-based defects (carpet) are harder. That's a more interesting and more honest finding than three categories all scoring 0.95+ would have been.

## What actually happened, category by category

### Bottle

This one went about as smoothly as training ever goes. Transfer learning off an ImageNet-pretrained ResNet50, frozen backbone at first, training just the final classification layer. That alone got to around 85% accuracy, but the confusion matrix told a different story than the headline number — the model was only catching about a third of real defects. It had learned that guessing "good" was usually a safe bet, because good bottles outnumbered defective ones roughly 3 to 1 in the data.

Fixed that with a class-weighted loss, which roughly doubled defect recall. Then I unfroze the last block of the backbone and fine-tuned it at a much lower learning rate than the head — the idea being that ImageNet's pretrained features were never trained to notice bottle-specific defects, so giving them a little room to adapt helped. That pushed defect recall to 0.91 and overall accuracy to 0.98.

### Cable

Cable has 8 defect types instead of bottle's 3, and a similar overall imbalance but way less data behind each individual defect type — some as few as 10 images. My first attempt at training this collapsed completely: the model just learned to predict "defect" for everything, presumably because the combination of class weighting and fine-tuning pushed it too hard, too fast, on noisy data. That was a real dead end, not a small tuning issue — the model was getting literally the same answer no matter what it saw.

Fixed it with a combination of things: softer class weights, gradient clipping to stop any single batch from throwing the model off course, a lower learning rate on the newly-initialized layer, and switching to saving whichever epoch had the best validation accuracy rather than just whatever the last epoch happened to be. That got training stable again. From there I tried a few different balancing techniques — reweighting the loss, using a weighted sampler to balance what the model actually sees during training, test-time augmentation, and focal loss. Most of them landed in a tight band around 89% accuracy, and one run of sampler-plus-plain-cross-entropy training landed at 92%, which is the version I kept.

Worth being upfront about: I didn't get 92% every single time I reran the exact same code. The sampler draws batches randomly, so identical code can give you a range of results depending on the random draw. 92% was a genuinely good run and I kept it, but I want to be honest that it wasn't the *only* number this setup produces.

### Carpet

This is where things got genuinely harder, and where I learned the most. Carpet's defects are texture-based (color variation, cuts, holes, contamination) rather than structural, and my early attempts landed the model in a lopsided spot — it was very good at catching real defects (95% recall) but at the cost of falsely flagging a lot of good carpet as defective too, which isn't a free trade in a real inspection system.

I tried softening the class balancing further to fix that, and it backfired badly — defect recall dropped to 40%, meaning the model started missing most real defects. That's a worse failure mode than the one I was trying to fix, so I reverted it.

Then I ran the same, reverted code three times in a row and got three different results: 0.84, 0.78, 0.75. Same code, same category, meaningfully different outcomes. That's when it became clear the randomness in the training process (mainly from the sampler) was swinging results by close to 10 percentage points on this category — carpet's test set is smaller than bottle's or cable's, so a handful of flipped predictions moves the number a lot.

Rather than keep rerunning and hoping for a better roll, I fixed a random seed so the whole training process became reproducible, then ran it once with that seed. It landed at 0.87 accuracy with a solid, balanced confusion matrix — not a fluke pulled from a pile of hidden reruns, but the first and only result from a seeded, reproducible run. Anyone who clones this repo and runs training with seed 42 should get this same result back.

## What I'd actually say this project demonstrates

Not "I trained a model that got 98% accuracy." More like: I can build a full pipeline, notice when a model's headline accuracy is hiding a real problem underneath (the imbalance issue), recognize when training has genuinely broken versus when it just needs patience (the cable collapse), know the difference between a real improvement and randomness dressed up as one (the carpet variance), and make a defensible call about when to stop tuning rather than chase a number indefinitely.

## Architecture

Raw MVTec images get split into stratified train/val/test sets, preserving the ratio of each defect type across splits. Images are loaded through a PyTorch `Dataset`, with augmentation (flips, rotation, color jitter) applied only during training — validation and test always see clean, unaugmented images so the numbers stay honest.

The model is a ResNet50 pretrained on ImageNet. For each category, the backbone starts frozen, with only a new classification head trained initially. From there, the last one or two residual blocks get unfrozen and fine-tuned at a much lower learning rate than the head, so the pretrained features can adapt without being wrecked by aggressive updates.

Class imbalance is handled with a `WeightedRandomSampler` that balances what the model sees during training, combined with gradient clipping and checkpointing on best validation accuracy — both added after the cable collapse taught me the training process needed more guardrails than I'd originally given it.

Every category gets its own processed CSVs, its own saved model checkpoint, and its own evaluation run — the whole pipeline is driven by a single `CATEGORY` environment variable, so adding a new category doesn't require touching any code, just dropping the raw images in the right folder.

## Setup

```powershell
git clone <repo-url>
cd VisionInspect-AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python albumentations timm fastapi uvicorn scikit-learn pandas pillow matplotlib python-multipart
```

Download the MVTec AD categories you want and drop them into `data/raw/<category>/`, keeping MVTec's own folder structure (`train/good/` and `test/<defect_type>/`).

## Running it

Set which category you're working with, then run the pipeline steps in order:

```powershell
$env:MVTEC_CATEGORY = "bottle"
python -m src.data.prepare_splits
python -m src.training.train
python -m src.evaluation.evaluate
```

To run every category found under `data/raw/` back to back, unattended:

```powershell
python -m src.run_all_categories
```

Single-image prediction from the command line:

```powershell
python -m src.inference.predict path\to\image.png
```

Or run the inference API and send it an image over HTTP:

```powershell
uvicorn src.api.main:app --reload
```
```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@path\to\image.png"
```

## Dataset

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) — currently trained on the `bottle`, `cable`, and `carpet` categories, out of 15 total in the full dataset.

## What's next

- The other 12 MVTec categories are sitting there ready to go — the pipeline handles them without code changes, it's really just a matter of time and, based on how carpet went, probably some individual attention for any other texture-based categories.
- A small frontend so you can upload a photo and see the prediction instead of using curl.
- Packaging the API in Docker.
- Multi-class prediction (which specific defect type, not just good/defect) is a bigger step up from here — worth exploring once the binary version is fully proven out.
