# Vendored XGrammar bridge

Swift bridge donor: ml-explore/mlx-swift-lm commit22157fc397b59acfb03e91c370bcbf2cfb10970e (MLXCXGrammar target only).
C++ source: XGrammar v0.1.30, upstream commit d476a48dcd8fa3b5afeddbe850e73bb3b1dcf505, retained under xgrammar/. LICENSE and NOTICE remain in that directory; picojson/DLPack retain embedded notices.

No guided-generation loop, completion/whitespace biases, tokenizer heuristics or prompt changes are imported. The local shim JSON Schema compiler explicitly sets strict_mode=false so omitted additionalProperties/items follow JSON Schema defaults. The Swift admission layer rejects unsupported schema constraints before compilation. Vendored C++ source is otherwise unchanged. The grammar_functor wrapper retains the donor static-member link fix.

The namespace macros isolate C++ xgrammar and picojson symbols. The existing package C++20 setting compiles the donor's C++17-compatible source; no global language-standard change is introduced.

Local C++20 compatibility: moved the SequenceFormat and OrFormat constructor bodies after all recursive variant alternatives in structural_tag.h. No grammar behavior change.

Local JSON key correctness fix: GetPropertyPattern first serializes the property name as a JSON string and then applies the existing JSONStrToPrintableStr EBNF quoting helper, matching VisitConst. This preserves quote, backslash, control and Unicode key semantics instead of inserting raw property names into EBNF literals. CPU regression: testPropertyNamesPreserveJSONEscapingAndUnicode.

Serialization policy: compile JSON schemas with any_whitespace=false, no indent, and explicit comma/colon separators. This restricts structural formatting at grammar compilation, without modifying string values, logits, sampling, or EOS. Donor generic basic collection rules retain their fixed comma-space separator; they have no unbounded structural whitespace loop.
