# Explicações geradas (qwen2.5:7b-instruct)

## Caso A — maligno mais óbvio (paciente 915143, real: MALIGNO)

**Resumo:** O modelo estima 100,0% de probabilidade de malignidade, faixa de risco ALTO.
**O que mais pesou:** (1) raio do núcleo (pior valor) de 30,79, muito maior que a mediana dos casos benignos (13,35), eleva muito o risco; (2) raio do núcleo (erro padrão) de 1,291, bem acima da mediana dos benignos (0,259), eleva o risco; (3) pontos côncavos (pior valor) de 0,226, acima da mediana dos benignos (0,074), contribui para o risco.
**Incerteza:** Embora a probabilidade seja muito alta, a consistência entre as medidas reduz a incerteza.
**Conduta sugerida:** Encaminhar com prioridade ao mastologista.
**Aviso:** Esta é uma ferramenta de apoio; a decisão clínica final é do médico.

## Caso B — benigno mais óbvio (paciente 891936, real: BENIGNO)

**Resumo:** A probabilidade estimada pelo modelo é 0,0%, com faixa de risco BAIXO.
**O que mais pesou:** (1) raio do núcleo (pior valor) de 11,37, bem abaixo da mediana dos casos benignos (13,35), reduz o risco; (2) textura (pior valor) de 14,82, também abaixo da mediana dos casos benignos (22,94), reduz o risco; (3) pontos côncavos (média) de 0,014, bem abaixo da mediana dos casos malignos (0,086), reduz o risco.
**Incerteza:** Embora a probabilidade seja baixa, a faixa de risco é BAIXO, indicando consistência entre as medidas.
**Conduta sugerida:** Seguir acompanhamento de rotina.
**Aviso:** Esta é uma ferramenta de apoio; a decisão clínica final é do médico.

## Caso C — mais ambíguo (perto de 50%) (paciente 879523, real: MALIGNO)

**Resumo:** A probabilidade estimada pelo modelo é de 41,1%, classificada como faixa MODERADO.
**O que mais pesou:** (1) raio do núcleo (pior valor) de 17,77, bem acima da mediana dos casos benignos (13,35), eleva o risco; (2) pontos côncavos (pior valor) de 0,125, também acima do típico benigno (0,074), eleva o risco; (3) simetria (pior valor) de 0,342, acima da mediana dos benignos (0,269), aumenta o risco.
**Incerteza:** A probabilidade está próxima do limiar, sendo considerada duvidosa e requerendo revisão humana.
**Conduta sugerida:** Exame de imagem com prioridade e revisão por um mastologista.
**Aviso:** Esta é uma ferramenta de apoio; a decisão clínica final é do médico.

## Caso D — pior erro (maligno com menor probabilidade) (paciente 859983, real: MALIGNO)

**Resumo:** A probabilidade estimada pelo modelo é de 23,9%, classificada como faixa BAIXO.
**O que mais pesou:** (1) raio do núcleo (pior valor) de 16,57, acima da mediana dos benignos (13,35), eleva o risco; (2) pontos côncavos (pior valor) de 0,138, também acima do típico benigno (0,074), eleva o risco; (3) pontos côncavos (média) de 0,051, acima do típico benigno (0,023), contribui para o risco.
**Incerteza:** A probabilidade é baixa, mas os fatores de risco são consistentes, sugerindo uma avaliação cuidadosa.
**Conduta sugerida:** Seguir acompanhamento de rotina conforme padrão.
**Aviso:** Esta é uma ferramenta de apoio; a decisão clínica final é do médico.
