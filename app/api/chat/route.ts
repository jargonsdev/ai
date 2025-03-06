import model from "@/lib/model";
import { LangChainAdapter } from "ai";
import { NextResponse } from "next/server";
import { formatMessage } from "@/lib/utils";
import vectorStore from "@/lib/vector-store";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { formatDocumentsAsString } from "langchain/util/document";
import { HttpResponseOutputParser } from "langchain/output_parsers";

export const dynamic = "force-dynamic";

const TEMPLATE = `Answer the user's questions based only on the following context. If the answer is not in the context, go ahead and provide answer by looking up your own knowledge directly; State a postscript of your response lightly to the user that the information though was not available in the context.:
------------------------------
Context: {context}
------------------------------
Current conversation: {chat_history}

user: {question}
jAI:`;

export async function OPTIONS() {
  return NextResponse.json(
    {},
    {
      status: 200,
      headers: {
        "Access-Control-Allow-Origin": process.env.ALLOWED_ORIGIN!,
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
      },
    },
  );
}

export async function POST(req: Request) {
  const corsHeaders = {
    "Access-Control-Allow-Origin": process.env.ALLOWED_ORIGIN!,
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
  };

  try {
    // Extract the `messages` from the body of the request
    const { messages } = await req.json();
    const formattedPreviousMessages = messages.slice(0, -1).map(formatMessage);
    const currentMessageContent = messages[messages.length - 1].content;

    // Create a prompt template
    const prompt = PromptTemplate.fromTemplate(TEMPLATE);

    // Get similar documents from the vector store
    const similarDocs = await vectorStore.similaritySearch(
      currentMessageContent,
    );

    // Create the parser - parses the response from the model into http-friendly format
    const parser = new HttpResponseOutputParser();

    // Create a chain of runnables - this is the core of the LangChain API
    const chain = RunnableSequence.from([
      {
        question: (input) => input.question,
        chat_history: (input) => input.chat_history,
        context: () => formatDocumentsAsString(similarDocs),
      },
      prompt,
      model,
      parser,
    ]);

    // Convert the response into a friendly text-stream
    const stream = await chain.stream({
      chat_history: formattedPreviousMessages.join("\n"),
      question: currentMessageContent,
    });

    // Convert Uint8Array stream to string stream before returning
    const textStream = new ReadableStream({
      async start(controller) {
        const decoder = new TextDecoder();
        for await (const chunk of stream) {
          controller.enqueue(decoder.decode(chunk));
        }
        controller.close();
      },
    });

    const response = LangChainAdapter.toDataStreamResponse(textStream);

    // Add CORS headers to the response
    Object.entries(corsHeaders).forEach(([key, value]) => {
      response.headers.set(key, value);
    });

    return response;
  } catch (e: any) {
    return Response.json(
      { error: e.message },
      {
        status: e.status ?? 500,
        headers: corsHeaders,
      },
    );
  }
}
