import type { Message } from 'ai';

/**
 * Formats a message into a string
 * @param message The message to format
 * @returns The formatted message
 */
export const formatMessage = (message: Message) => {
    return `${message.role}: ${message.content}`;
};