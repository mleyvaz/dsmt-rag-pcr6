# Resultados consolidados de DSm-RAG

## Alcance

Se ejecutaron dos experimentos sobre conjuntos RAG publicados:

- RAMDocs: 500 preguntas, ocho métodos y tres condiciones, con 12.000 predicciones.
- ConflictQA PopQA ChatGPT: 7.947 preguntas, seis métodos y cuatro condiciones, con 190.728 predicciones.

El experimento usa las respuestas documentales proporcionadas por los conjuntos como salida de un extractor perfecto de afirmaciones. Por tanto, mide la capa de fusión y no todavía la calidad de un extractor basado en LLM. En RAMDocs, el número y los nombres de las entidades desambiguadas se proporcionaron por igual a todos los métodos. Las etiquetas de documento se usaron únicamente para evaluación y para construir las condiciones de estrés declaradas.

## Resultado favorable principal en RAMDocs

En la condición que repite tres veces cada documento de desinformación, DSmT con PCR6 y descuento de procedencia obtuvo:

| Método | Exactitud del conjunto | F1 del conjunto | ECE |
|---|---:|---:|---:|
| Votación mayoritaria | 42,4% | 69,1% | 0,490 |
| DST factorizado | 46,4% | 71,9% | 0,428 |
| PCR6 sin procedencia | 46,0% | 71,8% | 0,347 |
| DST con procedencia | 78,2% | 91,0% | 0,160 |
| DSmT PCR6 con procedencia | **80,6%** | **92,3%** | 0,193 |

La diferencia de F1 entre DSmT PCR6 y DST, ambos con el mismo descuento de procedencia, fue de +1,33 puntos porcentuales. El intervalo bootstrap pareado del 95% fue [+0,60, +2,33]. En exactitud, DSmT obtuvo una diferencia de +2,4 puntos. Esta es la primera condición empírica del estudio donde la variante DSmT supera al control DST equiparado.

La calibración muestra un intercambio: DST con procedencia tuvo menor ECE que PCR6 en esta condición. Por tanto, la superioridad observada se refiere a recuperación del conjunto correcto, no a todas las métricas.

## RAMDocs original

En los 500 casos originales, DST factorizado obtuvo 75,4% de exactitud y F1 de 89,2%; DSmT PCR6 obtuvo 75,0% y F1 de 89,0%. La diferencia de F1 fue −0,13 puntos, con intervalo [−0,33, 0,00]. No existe evidencia de una mejora de exactitud de DSmT en la distribución original.

PCR6 sí presentó mejor calibración: ECE de 0,133 frente a 0,161 de DST factorizado. Este resultado justifica estudiar utilidad selectiva y abstención, donde una distribución de creencias menos extrema puede ser más valiosa que una diferencia mínima de exactitud.

## RAMDocs con desinformación dominante

DSmT PCR6 con procedencia obtuvo 67,6% de exactitud y F1 de 84,6%, mientras DST con procedencia obtuvo 67,8% y 85,0%. La diferencia de F1 fue −0,43 puntos y su intervalo [−1,93, +1,07] incluyó cero. Los métodos quedan estadísticamente empatados en F1.

DSmT volvió a mostrar mejor ECE, 0,132 frente a 0,156. La evidencia favorece una hipótesis de mejor moderación del conflicto, pero no una mayor recuperación de respuestas.

## ConflictQA

En la condición equilibrada, todos los métodos de evidencia eligieron correctamente la evidencia externa. DST alcanzó ECE de 0,327 y PCR6 de 0,394. No hubo ventaja para DSmT.

Cuando la memoria paramétrica falsa se repitió tres veces como copias de una misma procedencia, la votación, DST y PCR6 sin control de dependencia obtuvieron 0%. Al aplicar descuento de procedencia, tanto DST como DSmT recuperaron 100%. La mejora pertenece al modelado de dependencia, no a una regla específica.

Cuando las tres memorias falsas fueron tratadas como fuentes independientes, ningún método recuperó la respuesta correcta. Este control confirma que el algoritmo no utiliza las etiquetas de evaluación para decidir.

## Experimento con NLI automático

Se añadió `Xenova/distilbert-base-uncased-mnli` cuantizado a 8 bits. El modelo procesó 7.802 pares documento–afirmación de RAMDocs con longitud máxima de 256 tokens. Para cada par produjo probabilidades de entailment, neutral y contradiction. Las probabilidades fueron válidas y sumaron uno dentro del error numérico. Un caso no contenía ninguna respuesta documental candidata, por lo que 499 de los 500 casos generaron pares NLI.

La afirmación con mayor apoyo NLI de cada documento se convirtió en un soporte simple. La masa comprometida aumentó con entailment y disminuyó con neutralidad y contradicción; el resto se asignó a ignorancia. DST y DSmT recibieron exactamente esas mismas masas y el mismo agrupamiento de procedencia.

| Condición | Método | Exactitud | F1 | Brier | ECE |
|---|---|---:|---:|---:|---:|
| Original | NLI pooling | 54,0% | 79,3% | 0,421 | 0,422 |
| Original | DST + NLI | 59,4% | 80,9% | 0,350 | 0,295 |
| Original | DSmT PCR6 + NLI | **59,4%** | **80,9%** | **0,315** | **0,240** |
| Desinformación dominante | NLI pooling | 40,0% | 72,9% | 0,567 | 0,571 |
| Desinformación dominante | DST + NLI | 56,2% | 79,3% | 0,350 | 0,306 |
| Desinformación dominante | DSmT PCR6 + NLI | **56,2%** | **79,3%** | **0,331** | **0,266** |

PCR6 y Dempster seleccionaron las mismas respuestas, pero PCR6 asignó mejores probabilidades. En RAMDocs original, la diferencia pareada de Brier fue −0,0353 a favor de DSmT, con intervalo bootstrap de 95% [−0,0463, −0,0251]. Bajo desinformación dominante fue −0,0190, con intervalo [−0,0267, −0,0119]. Como todo el intervalo permanece por debajo de cero, la mejora de calibración es consistente en estas muestras.

Frente a NLI pooling, DSmT mejoró el F1 en 1,61 puntos en la distribución original, con intervalo [0,73, 2,53], y en 6,33 puntos bajo desinformación dominante, con intervalo [4,91, 7,78]. Esta comparación demuestra el valor de la capa estructurada de fusión, aunque no atribuye toda la mejora exclusivamente a PCR6 porque DST factorizado obtuvo las mismas decisiones.

## Conclusión que puede defenderse

Los resultados permiten formular una afirmación limitada y comprobable:

> En RAG con múltiples respuestas válidas y desinformación duplicada, una arquitectura DSmT que combina PCR6 con procedencia puede mejorar la recuperación del conjunto de respuestas frente a DST con el mismo control de dependencia. La ventaja no se reproduce en conflictos binarios simples y no implica mejor calibración en todas las condiciones.

La formulación evita afirmar que DSmT supera universalmente a Dempster–Shafer. También identifica el nicho experimental donde continuar: preguntas ambiguas con varias entidades, documentos parcialmente compatibles y evidencia falsa derivada de una misma fuente.

Con NLI, la conclusión puede ampliarse: PCR6 no cambió la exactitud frente a DST factorizado, pero redujo significativamente el error de Brier. La evidencia actual respalda una ventaja de calibración y gestión de incertidumbre, además de la mejora de recuperación observada en el experimento oráculo con desinformación duplicada.

## Próximo experimento

El NLI ya estima el apoyo de los documentos sin consultar las etiquetas `correct`, `misinfo` o `noise`. Todavía se utiliza el campo documental `answer` para enumerar respuestas candidatas y las entidades anotadas para estructurar el marco. La siguiente fase debe sustituir esos dos componentes por extracción libre de respuestas y vinculación automática de entidades. Después deben evaluarse tres decisiones adicionales: responder, entregar varias respuestas contextualizadas o abstenerse.
