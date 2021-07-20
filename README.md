# Sistema de apoio à decisão para aprovação de crédito

## Sobre o projeto
A ideia do projeto é identificar, dentre os clientes que solicitam um produto de crédito (como um cartão de crédito ou um empréstimo pessoal, por exemplo) e que cumprem os pré-requisitos essenciais para a aprovação do crédito, aqueles que apresentem alto risco de não conseguirem honrar o pagamento, tornando-se inadimplentes.

Para isso, foi utilizado um arquivo com dados históricos de 20.000 solicitações de produtos de créditos que foram aprovadas pela instituição, acompanhadas da indicação de quais desses solicitantes conseguiram honrar os pagamentos e quais ficaram inadimplentes.

Com base nesses dados históricos, foi construído um classificador que, a partir dos dados de uma nova solicitação de crédito, tenta predizer se este solicitante será um bom ou mau pagador.


## Tecnologias utilizadas
O projeto foi desenvolvido utilizando as seguintes tecnologias:

- Python
- Scikit-Learn
