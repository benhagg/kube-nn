{{- define "nanogpt.partitionKey" -}}
{{- printf "%s|%s|%d|%v|%s|%s" .Values.tokenizerImage .Values.tokenizer.encoding (int .Values.tokenizer.shardSize) .Values.tokenizer.validationFraction .Values.tokenizer.shardPrefix .Values.tokenizer.inputRevision | sha256sum -}}
{{- end -}}
