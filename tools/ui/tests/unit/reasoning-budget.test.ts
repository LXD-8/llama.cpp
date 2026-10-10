import { ReasoningEffort } from '$lib/enums';
import { ChatService } from '$lib/services/chat.service';
import { describe, expect, it } from 'vitest';

describe('reasoningBudgetTokens', () => {
	it('sends no budget by default', () => {
		expect(ChatService.reasoningBudgetTokens({})).toBe(-1);
		expect(ChatService.reasoningBudgetTokens({ enableThinking: false })).toBe(-1);
	});

	it('maps an effort level to its token budget when thinking is on', () => {
		expect(
			ChatService.reasoningBudgetTokens({
				enableThinking: true,
				reasoningEffort: ReasoningEffort.LOW
			})
		).toBe(512);
		expect(
			ChatService.reasoningBudgetTokens({
				enableThinking: true,
				reasoningEffort: ReasoningEffort.MAX
			})
		).toBe(-1);
	});

	it('turns thinking off with a zero budget, keeping the prompt unchanged', () => {
		expect(ChatService.reasoningBudgetTokens({ thinkingOffByBudget: true })).toBe(0);
	});
});
