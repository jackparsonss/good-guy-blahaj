import type { SiteConfig } from "@/types";

export const siteConfig: SiteConfig = {
	author: "blahaj",
	title: "HackEd 2024",
	description: "We hate hate speech",
	lang: "en-GB",
	ogLocale: "en_GB",
	date: {
		locale: "en-GB",
		options: {
			day: "numeric",
			month: "short",
			year: "numeric",
		},
	},
};

export const menuLinks: Array<{ title: string; path: string }> = [
	{
		title: "Realtime Speech Display",
		path: "/",
	},
];
