

import transformers
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
from datasets import ClassLabel, Sequence
import random
import pandas as pd
from IPython.display import display, HTML
from transformers import AutoTokenizer

"""You can find a script version of this notebook to fine-tune your model in a distributed fashion using multiple GPUs or TPUs [here](https://github.com/huggingface/transformers/tree/master/examples/question-answering).

# Fine-tuning de un modelo para la tarea de QA

En este cuaderno, veremos cómo hacer "fine-tuning" a uno de los modelos de [🤗 Transformers](https://github.com/huggingface/transformers) para la tarea de respuesta a una pregunta (QA), que es la tarea de extraer la respuesta a una pregunta de un contexto dado. Veremos cómo cargar fácilmente un conjunto de datos para este tipo de tareas y usar la API `Trainer` para ajustar un modelo en él.


** Nota: ** Este cuaderno afina los modelos que responden preguntas tomando una subcadena de un contexto, no generando texto nuevo.
"""

squad_v2 = False
model_checkpoint = "BSC-TeMU/roberta-base-bne"
batch_size = 16

"""## Cargando el dataset

Usaremos la biblioteca [🤗 Datasets](https://github.com/huggingface/datasets) para descargar los datos y obtener la métrica que necesitamos usar para la evaluación (para comparar nuestro modelo con el benchmark). Esto se puede hacer fácilmente con las funciones `load_dataset` y` load_metric`.
"""



"""Para nuestro ejemplo usaremos el [Dataset SQAC](https://huggingface.co/datasets/BSC-TeMU/SQAC). El cuaderno debe funcionar con cualquier conjunto de datos de respuesta a preguntas proporcionado por la biblioteca 🤗 Datasets. Si está utilizando su propio conjunto de datos definido a partir de un archivo JSON o csv (consulte la [documentación de Datasets[texto del enlace](https://)](https://huggingface.co/docs/datasets/loading_datasets.html#from-local-files) sobre cómo cargarlos ), es posible que necesite algunos ajustes en los nombres de las columnas utilizadas."""

datasets = load_dataset("BSC-TeMU/SQAC")

"""The `datasets` object itself is [`DatasetDict`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasetdict), which contains one key for the training, validation and test set."""

print(datasets)

print(datasets["train"][0])

"""Podemos ver que las respuestas están indicadas por su posición inicial en el texto (aquí en el carácter 473) y su texto completo, que es una subcadena del contexto como mencionamos anteriormente."""



def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "No puedes seleccionar más elementos que los que contiene el dataset"
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(lambda x: [typ.feature.names[i] for i in x])
    display(HTML(df.to_html()))

show_random_elements(datasets["train"])

"""## Preprocessing the training data"""


    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

import transformers
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

tokenizer("What is your name?", "My name is Sylvain.")

"""Dependiendo del modelo que seleccionó, verá diferentes claves en el diccionario devueltas por la celda de arriba. No importan mucho para lo que estamos haciendo aquí (solo sepa que son requeridos por el modelo que crearemos más adelante), puede obtener más información sobre ellos en [este tutorial](https://huggingface.co/transformers/preprocessing.html) si está interesado.

Ahora bien, una cosa específica para el preprocesamiento en cuestión es cómo tratar documentos muy largos. Por lo general, los truncamos en otras tareas, cuando son más largos que la longitud máxima de oración del modelo, pero aquí, eliminar parte del contexto puede resultar en la pérdida de la respuesta que estamos buscando. Para lidiar con esto, permitiremos que un ejemplo (largo) en nuestro conjunto de datos proporcione varias características de entrada, cada una de una longitud más corta que la longitud máxima del modelo (o la que establecemos como un hiperparámetro). Además, en caso de que la respuesta esté en el punto en que dividimos un contexto largo, permitimos cierta superposición entre las características que generamos controladas por el hiperparámetro `doc_stride`:
"""

max_length = 384 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.

"""Encontremos el ejemplo más largo en nuestro dataset:"""

for i, example in enumerate(datasets["train"]):
    if len(tokenizer(example["question"], example["context"])["input_ids"]) > 384:
        break
example = datasets["train"][i]

"""Sin truncar obtenemos la siguiente longitud de los input IDs:"""

len(tokenizer(example["question"], example["context"])["input_ids"])

"""Ahora, truncamos (y perdemos información):"""

len(tokenizer(example["question"], example["context"], max_length=max_length, truncation="only_second")["input_ids"])

"""Tenga en cuenta que nunca queremos truncar la pregunta, solo el contexto, por eso usamos el truncamiento `only_second`. Ahora, nuestro tokenizador puede devolvernos automáticamente una lista de características con un límite de cierta longitud máxima, con la superposición que hablamos """

tokenized_example = tokenizer(
    example["question"],
    example["context"],
    max_length=max_length,
    truncation="only_second",
    return_overflowing_tokens=True,
    stride=doc_stride
)

"""Ahora no tenemos una lista de `input_ids`, sino varias: """

[len(x) for x in tokenized_example["input_ids"]]

"""Y si los decodificamos, podemos ver la superposición:"""

for x in tokenized_example["input_ids"][:2]:
    print(tokenizer.decode(x))

"""Ahora, esto nos dará algo de trabajo para tratar adecuadamente las respuestas: necesitamos encontrar en cuál de esas características se encuentra realmente la respuesta y dónde exactamente en esa característica. Los modelos que usaremos requieren las posiciones inicial y final de estas respuestas en los tokens, por lo que también necesitaremos mapear partes del contexto original a algunos tokens. Afortunadamente, el tokenizador que estamos usando puede ayudarnos con eso devolviendo un `offset_mapping`:"""

tokenized_example = tokenizer(
    example["question"],
    example["context"],
    max_length=max_length,
    truncation="only_second",
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    stride=doc_stride
)
print(tokenized_example["offset_mapping"][0][:100])

"""Esto da, para cada índice de nuestro IDS de entrada, el carácter inicial y final correspondiente en el texto original que dio nuestro token. El primer token (`[CLS]`) tiene (0, 0) porque no corresponde a ninguna parte de la pregunta / respuesta, entonces el segundo token es el mismo que los caracteres 0 a 3 de la pregunta:"""

first_token_id = tokenized_example["input_ids"][0][1]
offsets = tokenized_example["offset_mapping"][0][1]
print(tokenizer.convert_ids_to_tokens([first_token_id])[0], example["question"][offsets[0]:offsets[1]])

"""Entonces, podemos usar este mapeo para encontrar la posición de los tokens de inicio y finalización de nuestra respuesta en una característica determinada. Solo tenemos que distinguir qué partes de las compensaciones corresponden a la pregunta y qué parte corresponden al contexto, aquí es donde el método `sequence_ids` de nuestro` tokenized_example` puede ser útil:"""

sequence_ids = tokenized_example.sequence_ids()
print(sequence_ids)

"""
Devuelve `None` para los tokens especiales, luego 0 o 1 dependiendo de si el token correspondiente proviene de la primera oración pasada (la pregunta) o de la segunda (el contexto). Ahora, con todo esto, podemos encontrar el primer y último token de la respuesta en una de nuestras funciones de entrada (o si la respuesta no está en esta función):"""

answers = example["answers"]
start_char = answers["answer_start"][0]
end_char = start_char + len(answers["text"][0])

# Start token index of the current span in the text.
token_start_index = 0
while sequence_ids[token_start_index] != 1:
    token_start_index += 1

# End token index of the current span in the text.
token_end_index = len(tokenized_example["input_ids"][0]) - 1
while sequence_ids[token_end_index] != 1:
    token_end_index -= 1

# Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
offsets = tokenized_example["offset_mapping"][0]
if (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
    # Move the token_start_index and token_end_index to the two ends of the answer.
    # Note: we could go after the last offset if the answer is the last word (edge case).
    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
        token_start_index += 1
    start_position = token_start_index - 1
    while offsets[token_end_index][1] >= end_char:
        token_end_index -= 1
    end_position = token_end_index + 1
    print(start_position, end_position)
else:
    print("The answer is not in this feature.")

"""Y podemos comprobar que es la respuesta correcta:"""

print(tokenizer.decode(tokenized_example["input_ids"][0][start_position: end_position+1]))
print(answers["text"][0])

"""Para que este cuaderno funcione con cualquier tipo de modelo, debemos tener en cuenta el caso especial en el que el modelo espera relleno a la izquierda (en cuyo caso cambiamos el orden de la pregunta y el contexto):"""

pad_on_right = tokenizer.padding_side == "right"

"""Ahora juntemos todo en una función que aplicaremos a nuestro conjunto de entrenamiento. En el caso de respuestas imposibles (la respuesta está en otra característica dada por un ejemplo con un contexto largo), establecemos el índice cls tanto para la posición inicial como para la final. También podríamos simplemente descartar esos ejemplos del conjunto de entrenamiento si la marca `allow_impossible_answers` es` False`. Dado que el preprocesamiento ya es lo suficientemente complejo como es, hemos mantenido que es simple para esta parte."""

def prepare_train_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

features = prepare_train_features(datasets['train'][:5])

tokenized_datasets = datasets.map(prepare_train_features, batched=True, remove_columns=datasets["train"].column_names)

"""## Fine-tuning del modelo"""



model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    f"{model_name}-finetuned-sqac",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

"""Luego, necesitaremos un `data_collator` que agrupe nuestros ejemplos procesados, aquí el predeterminado funcionará:"""

from transformers import default_data_collator

data_collator = default_data_collator

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

"""Como el entrenamiento es largo, guardemos el modelo por si necesitamos reiniciar el entrenamiento"""

trainer.save_model("test-squad-trained")

"""## Evaluación"""

def prepare_validation_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

"""And like before, we can apply that function to our validation set easily:"""

validation_features = datasets["test"].map(
    prepare_validation_features,
    batched=True,
    remove_columns=datasets["test"].column_names
)

"""Now we can grab the predictions for all features by using the `Trainer.predict` method:"""

raw_predictions = trainer.predict(validation_features)

validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))

max_answer_length = 30

start_logits = output.start_logits[0].cpu().numpy()
end_logits = output.end_logits[0].cpu().numpy()
offset_mapping = validation_features[0]["offset_mapping"]
# The first feature comes from the first example. For the more general case, we will need to be match the example_id to
# an example index
context = datasets["validation"][0]["context"]

# Gather the indices the best start/end logits:
start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
valid_answers = []
for start_index in start_indexes:
    for end_index in end_indexes:
        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
        # to part of the input_ids that are not in the context.
        if (
            start_index >= len(offset_mapping)
            or end_index >= len(offset_mapping)
            or offset_mapping[start_index] is None
            or offset_mapping[end_index] is None
        ):
            continue
        # Don't consider answers with a length that is either < 0 or > max_answer_length.
        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
            continue
        if start_index <= end_index: # We need to refine that test to check the answer is inside the context
            start_char = offset_mapping[start_index][0]
            end_char = offset_mapping[end_index][1]
            valid_answers.append(
                {
                    "score": start_logits[start_index] + end_logits[end_index],
                    "text": context[start_char: end_char]
                }
            )

valid_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[:n_best_size]
valid_answers

datasets["test"][0]["answers"]

import collections

examples = datasets["test"]
features = validation_features

example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
features_per_example = collections.defaultdict(list)
for i, feature in enumerate(features):
    features_per_example[example_id_to_index[feature["example_id"]]].append(i)

from tqdm.auto import tqdm

def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size = 20, max_answer_length = 30):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []
        
        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}
        
        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        if not squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer

    return predictions

"""And we can apply our post-processing function to our raw predictions:"""

final_predictions = postprocess_qa_predictions(datasets["test"], validation_features, raw_predictions.predictions)

"""Then we can load the metric from the datasets library."""

metric = load_metric("squad")

formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["test"]]
metric.compute(predictions=formatted_predictions, references=references)

"""Ahora puede cargar el resultado del entrenamiento en el Hub, simplemente ejecute esta instrucción:"""
