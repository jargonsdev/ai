import { LangChainAdapter } from 'ai';
import { ChatOpenAI } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';
import { HttpResponseOutputParser } from 'langchain/output_parsers';

import { formatMessage } from '@/lib/utils';
import data from '@/data/dictionary.json';

import { JSONLoader } from "langchain/document_loaders/fs/json";
import { RunnableSequence } from '@langchain/core/runnables'
import { formatDocumentsAsString } from 'langchain/util/document';

const loader = new JSONLoader(
    "../../data/dictionary.json",
    ["title", "content"],
);

export const dynamic = 'force-dynamic'

const TEMPLATE = `Answer the user's questions based only on the following context. If the answer is not in the context, go ahead and provide answer by looking up your own knowledge directly; State a postscript of your response lightly to the user that the information though was not available in the context.:
------------------------------
Context: {context}
------------------------------
Current conversation: {chat_history}

user: {question}
jAI:`;

export async function POST(req: Request) {
    try {
        // Extract the `messages` from the body of the request
        const { messages } = await req.json();
        const formattedPreviousMessages = messages.slice(0, -1).map(formatMessage);
        const currentMessageContent = messages[messages.length - 1].content;

        // Load the documents
        const docs = await loader.load();

        // Create a prompt template
        const prompt = PromptTemplate.fromTemplate(TEMPLATE);

        // Create the model
        const model = new ChatOpenAI({
            apiKey: process.env.OPENAI_API_KEY!,
            model: 'gpt-3.5-turbo',
            temperature: 0,
            streaming: true,
            verbose: true,
        });

        // Create the parser - parses the response from the model into http-friendly format
        const parser = new HttpResponseOutputParser();

        // Create a chain of runnables - this is the core of the LangChain API
        const chain = RunnableSequence.from([
            {
                question: (input) => input.question,
                chat_history: (input) => input.chat_history,
                context: () => formatDocumentsAsString(docs),
            },
            prompt,
            model,
            parser,
        ]);

        // Convert the response into a friendly text-stream
        const stream = await chain.stream({
            chat_history: formattedPreviousMessages.join('\n'),
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

        return LangChainAdapter.toDataStreamResponse(textStream);
    } catch (e: any) {
        return Response.json({ error: e.message }, { status: e.status ?? 500 });
    }
}