[Prompt]
{{prompt}}
(end of [Prompt])

[Note]
I have made an effort to remove personally identifying information from the essays using the Named Entity Recognizer (NER). The relevant entities are identified in the text and then replaced with a string such as "{PERSON}", "{ORGANIZATION}", "{LOCATION}", "{DATE}", "{TIME}", "{MONEY}", "{PERCENT}”, “{CAPS}” (any capitalized word) and “{NUM}” (any digits). Please do not penalize the essay because of the anonymizations.
(end of [Note])

[Essay]
{{essay}}
(end of [Essay])

Q. List the quotations from the [Essay] that are relevant to “{{trait}}” and evaluate whether each quotation is well-written or not.