---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:4950
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: Dark Match Action Horror 2024 A small-time wrestling company accepts
    a well-paying gig in a backwoods town only to learn, too late, that the community
    is run by a mysterious cult leader with devious plans for their match.
  sentences:
  - Bring Them Down Drama Thriller 2025 When the ongoing rivalry between farmers Michael
    and Jack suddenly escalates, it triggers a chain of events that take increasingly
    violent and devastating turns, leaving both families permanently altered.
  - The Shawshank Redemption Drama Crime 1994 Imprisoned in the 1940s for the double
    murder of his wife and her lover, upstanding banker Andy Dufresne begins a new
    life at the Shawshank prison, where he puts his accounting skills to work for
    an amoral warden. During his long stretch in prison, Dufresne comes to be admired
    by the other inmates -- including an older prisoner named Red -- for his integrity
    and unquenchable sense of hope.
  - Presence Horror Drama Thriller 2025 A couple, Rebekah and Chris, and their children,
    Tyler and Chloe, move into a seemingly normal suburban home. When strange occurrences
    occur, they begin to believe that there is something else in the house with them.
    The presence is about to disrupt their lives in unimaginable ways.
- source_sentence: 'Captain America: Brave New World Action Thriller Science Fiction
    2025 After meeting with newly elected U.S. President Thaddeus Ross, Sam finds
    himself in the middle of an international incident. He must discover the reason
    behind a nefarious global plot before the true mastermind has the entire world
    seeing red.'
  sentences:
  - Escape Action Drama 2024 After completing his required decade of military service
    and being honored as a hero, a North Korean sergeant makes a sudden shocking attempt
    to defect to the South, risking life and limb for the chance to finally determine
    his own destiny.
  - The Monkey Horror Comedy 2025 When twin brothers find a mysterious wind-up monkey,
    a series of outrageous deaths tear their family apart. Twenty-five years later,
    the monkey begins a new killing spree forcing the estranged brothers to confront
    the cursed toy.
  - María, ¡Me muero! Comedy 2024 “Maria, I'm dying!” is a comedy about a hypochondriac
    man, the only being in the world capable of even terminal colds, and his wife
    who has to put up with it until he decides to do something about it. The fear
    of death has never been so fun.
- source_sentence: Moana 2 Animation Adventure Family Comedy 2024 After receiving
    an unexpected call from her wayfinding ancestors, Moana journeys alongside Maui
    and a new crew to the far seas of Oceania and into dangerous, long-lost waters
    for an adventure unlike anything she's ever faced.
  sentences:
  - 'Venom: The Last Dance Action Science Fiction Adventure 2024 Eddie and Venom are
    on the run. Hunted by both of their worlds and with the net closing in, the duo
    are forced into a devastating decision that will bring the curtains down on Venom
    and Eddie''s last dance.'
  - Absolution Action Crime Thriller Mystery 2024 An aging ex-boxer gangster working
    as muscle for a Boston crime boss receives an upsetting diagnosis.  Despite a
    faltering memory, he attempts to rectify the sins of his past and reconnect with
    his estranged children. He is determined to leave a positive legacy for his grandson,
    but the criminal underworld isn’t done with him and won’t loosen their grip willingly.
  - Peter Pan's Neverland Nightmare Horror Thriller Fantasy 2025 Wendy Darling strikes
    out in an attempt to rescue her brother Michael from the clutches of the evil
    Peter Pan who intends to send him to Neverland. Along the way she meets a twisted
    Tinkerbell, who is hooked on what she thinks is fairy dust.
- source_sentence: Frogman Horror Fantasy 2024 An amateur filmmaker, struggling to
    turn his passion into a career, returns with friends to Loveland, Ohio, the location
    of his first, notorious sighting of the Frogman, determined to obtain irrefutable
    proof that the cryptid legend exists.
  sentences:
  - Alexander and the Terrible, Horrible, No Good, Very Bad Road Trip Comedy Family
    2025 Eleven-year-old Alexander and his family embark on a dream Spring Break vacation
    to Mexico City only to have all their plans go terribly wrong when they discover
    a cursed idol.
  - 'Mesa de regalos Comedy 2025 Nicolás and Antonia are two inseparable friends who
    are far from the emotional and professional stability that all their acquaintances
    have achieved. After attending countless weddings, they come up with a plan as
    bold as it is ingenious: to fund their dreams by organizing their own fake wedding!
    They announce their "secret love," plan a big celebration, and at the crucial
    moment, Antonia is to leave Nicolás standing at the altar. Confident that the
    scandal will discourage guests from claiming their gifts, they plan to split the
    loot. However, unexpected feelings begin to arise, which could complicate their
    brilliant plan.'
  - Love Hurts Action Comedy Romance 2025 A realtor is pulled back into the life he
    left behind after his former partner-in-crime resurfaces with an ominous message.
    With his crime-lord brother also on his trail, he must confront his past and the
    history he never fully buried.
- source_sentence: Ask Me What You Want Romance Drama 2024 After his father's death,
    Eric Zimmerman travels to Spain to oversee his company's branches. In Madrid,
    he falls for Judith and engage in an intense, erotic relationship full of exploration.
    However, Eric fears his secret may end their affair.
  sentences:
  - Operation Undead Horror Action Thriller 2024 Inexperienced Thai soldiers battle
    a growing undead menace in this gruesome survival horror. A Japanese military
    experiment that turns men into monsters escapes containment, it’s up to these
    troops to save their nation from annihilation.
  - Heretic Horror Thriller 2024 Two young missionaries are forced to prove their
    faith when they knock on the wrong door and are greeted by a diabolical Mr. Reed,
    becoming ensnared in his deadly game of cat-and-mouse.
  - The Vigilante Thriller Action 2023 Returning from Afghanistan, Jessica, a Spec
    OPS Marine, finds herself in a war she never imagined and discovers middle America
    suburbia has changed when her thirteen year old sister, Aimee, is abducted by
    sex traffickers.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    "Ask Me What You Want Romance Drama 2024 After his father's death, Eric Zimmerman travels to Spain to oversee his company's branches. In Madrid, he falls for Judith and engage in an intense, erotic relationship full of exploration. However, Eric fears his secret may end their affair.",
    'The Vigilante Thriller Action 2023 Returning from Afghanistan, Jessica, a Spec OPS Marine, finds herself in a war she never imagined and discovers middle America suburbia has changed when her thirteen year old sister, Aimee, is abducted by sex traffickers.',
    'Heretic Horror Thriller 2024 Two young missionaries are forced to prove their faith when they knock on the wrong door and are greeted by a diabolical Mr. Reed, becoming ensnared in his deadly game of cat-and-mouse.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 4,950 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         | label                                                            |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | float                                                            |
  | details | <ul><li>min: 7 tokens</li><li>mean: 62.94 tokens</li><li>max: 188 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 64.69 tokens</li><li>max: 188 tokens</li></ul> | <ul><li>min: 0.07</li><li>mean: 0.34</li><li>max: 0.56</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                             | sentence_1                                                                                                                                                                                                                                                                                                                                                              | label                            |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------|
  | <code>Uppercut Drama Thriller 2025 When Elliott, a tough ex-boxing champion, accepts the challenge to train Toni, the two mismatched characters form an unlikely alliance. Their sparring and Elliott's keen insights show the resilient young fighter that real strength comes from the challenges you overcome when life throws its biggest punches your way.</code> | <code>xXx Action Adventure Thriller Crime 2002 Xander Cage is your standard adrenaline junkie with no fear and a lousy attitude. When the US Government "recruits" him to go on a mission, he's not exactly thrilled. His mission: to gather information on an organization that may just be planning the destruction of the world, led by the nihilistic Yorgi.</code> | <code>0.4047646155823832</code>  |
  | <code>I, the Executioner Action Crime 2024 The veteran detective Seo Do-cheol and his team at Major Crimes, relentless in their pursuit of criminals, join forces with rookie cop Park Sun-woo to track down a serial killer who has plunged the nation into turmoil.</code>                                                                                           | <code>The Vigilante Thriller Action 2023 Returning from Afghanistan, Jessica, a Spec OPS Marine, finds herself in a war she never imagined and discovers middle America suburbia has changed when her thirteen year old sister, Aimee, is abducted by sex traffickers.</code>                                                                                           | <code>0.35230944908183554</code> |
  | <code>Plankton: The Movie Animation Adventure Comedy Family Fantasy 2025 Plankton's tangled love story with his sentient computer wife goes sideways when she takes a stand — and decides to destroy the world without him.</code>                                                                                                                                     | <code>Sex Education Mistresses Comedy 1973 An erotic comedy depicting the sex adventures of  young wives living in an apartment complex, that are not satisfied with just having sex with their husbands.</code>                                                                                                                                                        | <code>0.3083212971687317</code>  |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 2
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `tp_size`: 0
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 1.6129 | 500  | 0.0019        |


### Framework Versions
- Python: 3.11.12
- Sentence Transformers: 3.4.1
- Transformers: 4.50.3
- PyTorch: 2.6.0+cu124
- Accelerate: 1.5.2
- Datasets: 3.5.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->