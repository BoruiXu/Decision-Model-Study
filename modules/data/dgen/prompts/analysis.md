The model makes one kind of mistake: {kind}. Below are real records from one group of these mistakes, each a JSON object of path to value, with the model's probability for its wrong decision.

{records}

Medians over records of the same true label that the model decides correctly:
{med_ok}

Medians over records of the label the model wrongly chose, decided correctly:
{med_other}

Work out how this group differs from the correctly decided records. Answer with {{"analysis": "<why the model gets these records wrong>", "pattern": "<what the records of this group have in common>", "strategy": "<how to write new records of this group that would teach the model: which fields to change, and within which ranges>"}}.
