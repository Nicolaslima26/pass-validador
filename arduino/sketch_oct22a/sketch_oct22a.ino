int ledPin = 8;            // LED normal
int ledPinvermelho = 9;    // LED vermelho
int ledPinAzul = 10;       // LED azul

void setup() {
  // Inicializa os pinos dos LEDs como saída
  pinMode(ledPin, OUTPUT);
  pinMode(ledPinvermelho, OUTPUT);
  pinMode(ledPinAzul, OUTPUT);

  // Inicializa a comunicação serial
  Serial.begin(9600);
}

void loop() {

  // Verifica se há dados disponíveis na serial
  if (Serial.available() > 0) {
    char command = Serial.read();  // Lê o comando enviado pelo Python

    // Se o comando for '1', acende o LED normal e apaga os outros
    if (command == '1') {
      digitalWrite(ledPin, HIGH);
      digitalWrite(ledPinvermelho, LOW);
      digitalWrite(ledPinAzul, LOW);
    }
    // Se o comando for '0', acende o LED vermelho e apaga os outros
    else if (command == '0') {
      digitalWrite(ledPin, LOW);
      digitalWrite(ledPinvermelho, HIGH);
      digitalWrite(ledPinAzul, LOW);
    }
    // Se o comando não for '1' nem '0', acende o LED azul e apaga os outros
    else {
      digitalWrite(ledPin, LOW);
      digitalWrite(ledPinvermelho, LOW);
      digitalWrite(ledPinAzul, HIGH);
    }
  }
}
