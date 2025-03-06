import { ChatOpenAI } from "@langchain/openai";

// Create the model
const model = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
  model: process.env.OPENAI_CHAT_MODEL!,
  temperature: 0,
  streaming: true,
  verbose: true,
});

export { model as default };
