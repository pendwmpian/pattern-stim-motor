/*
  Arduino LED Control via Serial
  This sketch listens for "ON" or "OFF" messages on the serial port
  and controls the state of the LED connected to pin 7 accordingly.
*/

const int ledPin = 7; // The pin the LED is connected to

void setup() {
  // Initialize the serial communication at 9600 bits per second:
  Serial.begin(9600);
  // Initialize the digital pin as an output.
  pinMode(ledPin, OUTPUT);
  // Set the initial state of the pin to LOW (off)
  digitalWrite(ledPin, LOW);
  Serial.println("Arduino is ready. Send 'ON' or 'OFF'.");
}

void loop() {
  // Check if data is available to read from the serial port
  if (Serial.available() > 0) {
    // Read the incoming string until a newline character is received
    String message = Serial.readStringUntil('\n');

    // Trim whitespace from the message
    message.trim();

    // Check the content of the message
    if (message == "ON") {
      // If the message is "ON", set the pin LOW (turn LED on, assuming common anode or standard wiring)
      digitalWrite(ledPin, LOW);
      Serial.println("LED is ON");
    } else if (message == "OFF") {
      // If the message is "OFF", set the pin HIGH (turn LED off)
      digitalWrite(ledPin, HIGH);
      Serial.println("LED is OFF");
    } else {
      Serial.print("Unknown command: ");
      Serial.println(message);
    }
  }
}
